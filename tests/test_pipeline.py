"""End-to-end: the pipeline runs, and the stream matches the batch path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.build import build_datasets
from data.datasets import ClipDataset, FeatureConfig, WindowDataset, collate_windows
from data.labels import LabelConfig
from data.skeleton import NUM_JOINTS
from data.stream import SkeletonStream, compare_offline
from data.synthetic import make_dataset
from evaluation.decision import DecisionConfig
from evaluation.metrics import evaluate
from evaluation.runner import score_dataset
from losses.preimpact import PreImpactLoss, PreImpactLossConfig
from models.build import build_model


@pytest.fixture(scope="module")
def clips():
    return make_dataset(num_subjects=6, falls_per_subject=2, adls_per_subject=2, seed=0)


def test_clip_record_round_trips(tmp_path, clips):
    original = clips[0]
    from data.clips import ClipRecord

    restored = ClipRecord.load(original.save(tmp_path))
    assert restored.clip_id == original.clip_id
    assert restored.impact_frame == original.impact_frame
    assert restored.fps == original.fps
    assert np.array_equal(restored.keypoints, original.keypoints)


def test_fall_clip_requires_an_impact_frame():
    from data.clips import ClipRecord

    with pytest.raises(ValueError, match="impact_frame"):
        ClipRecord(
            clip_id="x",
            subject="S00",
            keypoints=np.zeros((50, NUM_JOINTS, 3), dtype=np.float32),
            label="fall",
        )


def test_window_and_clip_datasets_agree_on_channels(clips):
    features_cfg = FeatureConfig(window=30, stride=5)
    windows = WindowDataset(clips, features_cfg, LabelConfig())
    stack = ClipDataset(clips, features_cfg, LabelConfig())[0]["windows"]
    assert windows[0]["features"].shape == (4, 30, NUM_JOINTS)
    assert stack.shape[1:] == (4, 30, NUM_JOINTS)


def test_no_velocity_gives_two_channels(clips):
    features_cfg = FeatureConfig(window=30, stride=10, with_velocity=False)
    assert features_cfg.in_channels == 2
    assert WindowDataset(clips, features_cfg, LabelConfig())[0]["features"].shape[0] == 2


def test_score_stream_aligns_with_the_clip(clips):
    dataset = ClipDataset(clips[:4], FeatureConfig(window=30), LabelConfig())
    model = build_model({"name": "stgcn", "blocks": [[32, 1], [32, 1]], "kernel_size": 5},
                        in_channels=4)
    scored = score_dataset(model, dataset, torch.device("cpu"))

    assert len(scored) == 4
    for item in scored:
        assert item.scores.shape == (item.clip.num_frames,)
        assert np.all((item.scores >= 0.0) & (item.scores <= 1.0))


def _fixed_scale_clip(num_frames: int = 80, fps: float = 30.0):
    """A clip on which the online and offline feature paths must agree exactly.

    Two nuisance differences are removed by construction rather than by loosening
    a tolerance:

    * **Constant torso length.** `_pose` places the shoulder centre exactly
      `torso` away from the hip centre whatever the lean, so holding `torso`
      fixed makes the offline clip-median scale and the online running-median
      scale identical numbers. The pose itself still moves freely.
    * **A still first frame.** Offline back-fills `velocity[0]` from
      `velocity[1]`; online has no previous frame and emits zero. Repeating
      frame 0 makes both zero.

    What is left is the score path, which is what this test is for.
    """
    from data.clips import ClipRecord
    from data.synthetic import _pose

    keypoints = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)
    lean = np.concatenate([[0.0], np.linspace(0.0, 1.2, num_frames - 1)])

    for t in range(num_frames):
        keypoints[t, :, :2] = _pose(
            root=np.array([300.0 + 0.7 * t, 250.0], dtype=np.float32),
            trunk_angle=float(lean[t]),
            torso=110.0,                       # constant: see the docstring
            hip_flex=0.3 * float(np.sin(t / 7.0)),
            knee_flex=0.2 + 0.4 * float(lean[t]),
            arm_swing=0.3 * float(np.cos(t / 5.0)),
            arm_raise=0.0,
        )
    # Deliberately noise-free: keypoint jitter would perturb the measured torso
    # length frame by frame, and the running median over the first window would
    # then differ from the clip median by a hair - reintroducing exactly the
    # nuisance this fixture removes. The pose still varies over time.
    keypoints[0] = keypoints[1]                # still first frame
    keypoints[:, :, 2] = 1.0                   # full confidence: no gap-filling
    return ClipRecord(clip_id="fixed", subject="S00", keypoints=keypoints, fps=fps)


def test_streaming_scores_match_the_batch_path():
    """The live path and the offline path must produce the same score stream.

    If these diverge, the reported metrics do not describe the system that would
    actually be deployed - the offline number would be measuring a model that
    sees features the live one never gets.
    """
    clip = _fixed_scale_clip()
    model = build_model({"name": "stgcn", "blocks": [[32, 1], [32, 1]], "kernel_size": 5},
                        in_channels=4).eval()
    batch = score_dataset(
        model, ClipDataset([clip], FeatureConfig(window=30), LabelConfig()),
        torch.device("cpu"),
    )[0].scores

    stream = SkeletonStream(window=30, fps=clip.fps)
    online = []
    for t in range(clip.num_frames):
        window = stream.push(clip.keypoints[t])
        if window is None:
            continue
        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1)
            online.append(float(torch.sigmoid(model(x))[0, -1]))

    assert np.allclose(batch[-len(online):], online, atol=1e-5)


def test_online_offline_feature_gap_is_bounded(clips):
    """On real clips the two paths differ, and by how much is a reportable number.

    Offline interpolates a dropped joint from both sides of the gap and scales by
    the clip's median torso length; online can do neither. Positions stay very
    close; velocity is where the gap shows up, because holding a joint still
    through a dropout and then jumping when it returns produces a spike that
    two-sided interpolation smooths away. The bound here is loose enough to be a
    regression guard, not a claim - `compare_offline` is what reports the actual
    figure for the paper.
    """
    for clip in clips[:5]:
        gap = compare_offline(clip, window=30)
        assert gap["frames_compared"] > 0
        assert gap["mean_abs_diff_position"] < 0.01, (
            "online and offline positions have diverged materially"
        )
        assert gap["mean_abs_diff_velocity"] < 0.5


def test_a_short_training_run_reduces_the_loss(clips):
    """Not an accuracy claim - just that gradients reach the whole model."""
    cfg = {
        "run": {"seed": 0},
        "data": {"split": {"mode": "single", "seed": 0}},
        "features": {"window": 30, "stride": 10},
        "labels": {"w_pre": 20},
        "augment": {"enabled": False},
    }
    bundle = build_datasets(cfg, clips)
    model = build_model({"name": "stgcn", "blocks": [[32, 1], [32, 1]], "kernel_size": 5},
                        in_channels=4)
    criterion = PreImpactLoss(PreImpactLossConfig(lam=1.5))
    criterion.set_pos_weight_from_prior(bundle.train.positive_fraction())
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = torch.utils.data.DataLoader(
        bundle.train, batch_size=16, shuffle=True, collate_fn=collate_windows
    )
    losses = []
    for epoch in range(3):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            optimiser.zero_grad(set_to_none=True)
            loss, _ = criterion(model(batch["features"]), batch["labels"], batch["tti"])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss)
            batches += 1
        losses.append(epoch_loss / batches)

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    scored = score_dataset(model, bundle.test_clips, torch.device("cpu"))
    report = evaluate(scored, DecisionConfig(), w_pre=20)
    assert 0.0 <= report.recall <= 1.0
    assert report.num_falls > 0


def test_ablation_configs_build_distinct_models():
    """Each ablation must actually differ from the full model."""
    from utils.config import load_config

    full = load_config("configs/ours_preimpact.yaml")
    variants = {
        "no_temporal": load_config("configs/ablations/no_temporal.yaml"),
        "no_velocity": load_config("configs/ablations/no_velocity.yaml"),
        "no_preimpact_loss": load_config("configs/ablations/no_preimpact_loss.yaml"),
        "no_grounding": load_config("configs/ablations/no_grounding.yaml"),
    }
    for name, cfg in variants.items():
        differences = [
            key for key in ("model", "loss", "features")
            if cfg[key] != full[key]
        ]
        assert differences, f"ablation {name} is identical to the full model"

    assert variants["no_temporal"]["model"]["temporal_pool"] is True
    assert variants["no_velocity"]["features"]["with_velocity"] is False
    assert variants["no_preimpact_loss"]["loss"]["lam"] == 0.0
    assert variants["no_grounding"]["model"]["attention"] is False


def test_receptive_field_fits_the_default_window():
    from models.stgcn import STGCNConfig
    from utils.config import load_config

    cfg = load_config("configs/default.yaml")
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model_cfg["blocks"] = tuple(tuple(b) for b in model_cfg["blocks"])
    assert STGCNConfig(**model_cfg).receptive_field <= cfg["features"]["window"]
