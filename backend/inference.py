"""Model loading and analysis for the demo API.

Everything here routes through the *same* code path as `eval.py` - the same
`ClipDataset` windowing, the same `evaluation.runner.score_clip` frame
alignment, the same `evaluation.decision` trigger rule. The demo is not a
reimplementation with its own conventions; if it disagreed with the reported
metrics, one of the two would be wrong and there would be no way to tell which.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.clips import ClipRecord, load_cache  # noqa: E402
from data.datasets import ClipDataset, FeatureConfig  # noqa: E402
from data.labels import LabelConfig  # noqa: E402
from data.skeleton import BONES, FLIP_PAIRS, JOINT_NAMES, NUM_JOINTS  # noqa: E402
from evaluation.decision import DecisionConfig, all_triggers  # noqa: E402
from evaluation.runner import score_clip  # noqa: E402
from explain.faithfulness import faithfulness_curves  # noqa: E402
from explain.relevance import joint_relevance  # noqa: E402
from explain.report import locate_window  # noqa: E402
from models.build import build_model  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).parent / "model" / "best.pt"
DEMO_CLIPS_DIR = Path(__file__).parent / "demo_clips"

# Render's free tier is one shared CPU with 512 MB, and the memory budget is far
# tighter than it looks: importing torch costs 403 MB and the checkpoint another
# 9, so ~100 MB is left for every forward pass the service will ever run.
#
# Scoring a clip in one batch does not fit. Peak RSS on a 141-frame clip, by
# chunk size (measured, reference machine):
#
#     chunk   peak     vs 512 MB    latency
#       112   681 MB   OOM, killed   2464 ms   <- the original default
#        32   521 MB   OOM, killed   2096 ms
#        16   469 MB   +43 MB        2081 ms
#         8   449 MB   +63 MB        1892 ms
#
# Chunking is not a speed/memory trade here - the small chunks are *faster*,
# because the working set stays in cache. 8 is chosen for headroom on a shared
# container, not because 16 was too slow. Raise it on an instance with real
# memory; there is no accuracy implication either way, since the chunks are
# independent forward passes over the same windows.
CHUNK_SIZE = max(1, int(os.environ.get("INFERENCE_CHUNK_SIZE", "8")))

# A cap keeps a large upload from turning into a request that outlives the
# platform's timeout - or exhausts what little memory remains.
MAX_FRAMES = max(60, int(os.environ.get("MAX_FRAMES", "600")))


@dataclass
class LoadedModel:
    """A checkpoint plus everything derived from its config."""

    model: torch.nn.Module
    config: dict
    decision: DecisionConfig
    features: FeatureConfig
    labels: LabelConfig
    device: torch.device
    checkpoint_path: str
    epoch: int | None

    @property
    def info(self) -> dict:
        return {
            "backbone": self.config["model"].get("name", "stgcn"),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "in_channels": self.features.in_channels,
            "window": self.features.window,
            "w_pre_frames": self.labels.w_pre,
            "with_velocity": self.features.with_velocity,
            "attention": bool(self.config["model"].get("attention", True)),
            "causal": bool(self.config["model"].get("causal", True)),
            "explain_method": self.config.get("explain", {}).get("method", "attention"),
            "trained_on": self.config.get("data", {}).get("source", "unknown"),
            "epoch": self.epoch,
            "device": str(self.device),
            "checkpoint": self.checkpoint_path,
        }


@lru_cache(maxsize=1)
def load_model(checkpoint: str | None = None) -> LoadedModel:
    """Load the demo checkpoint once per process and cache it."""
    path = Path(checkpoint or DEFAULT_CHECKPOINT)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Train one and copy it there:\n"
            f"  python train.py --config configs/ours_preimpact.yaml\n"
            f"  copy results/<run>/best.pt backend/model/best.pt"
        )

    device = torch.device("cpu")   # Render free tier has no GPU
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]

    model = build_model(dict(cfg["model"]), in_channels=int(blob.get("in_channels", 4)))
    model.load_state_dict(blob["model"])
    model.eval().to(device)
    # Render's free tier is a fraction of a shared core, where torch's default
    # thread pool spends more time coordinating than computing on a model this
    # small - so default to 1. Local development has real cores; set
    # TORCH_NUM_THREADS to use them.
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))

    stored = blob.get("decision") or cfg["decision"]
    return LoadedModel(
        model=model,
        config=cfg,
        decision=DecisionConfig(
            threshold=float(stored["threshold"]),
            persistence=int(stored["persistence"]),
            refractory_frames=int(stored["refractory_frames"]),
        ),
        features=FeatureConfig(**dict(cfg["features"])),
        labels=LabelConfig(**dict(cfg["labels"])),
        device=device,
        checkpoint_path=str(path),
        epoch=blob.get("epoch"),
    )


def skeleton_spec() -> dict:
    """Joint names, bones and mirror pairs, so the client draws what we model."""
    return {
        "joints": list(JOINT_NAMES),
        "bones": [list(b) for b in BONES],
        "flip_pairs": [list(p) for p in FLIP_PAIRS],
        "num_joints": NUM_JOINTS,
    }


def _as_clip(
    keypoints: np.ndarray, fps: float, impact_frame: int | None, clip_id: str
) -> ClipRecord:
    """Wrap a raw keypoint array as a `ClipRecord`.

    A sequence with no known impact frame is an ADL clip as far as the metrics
    are concerned - not a fall with an unknown `t*`. Lead time is simply
    undefined for it, and reporting one would mean inventing a `t*`.
    """
    return ClipRecord(
        clip_id=clip_id,
        subject="demo",
        keypoints=keypoints,
        fps=float(fps),
        label="fall" if impact_frame is not None else "adl",
        impact_frame=impact_frame,
        activity="unknown",
        source="api",
    )


def _explain(
    loaded: LoadedModel, item: dict, frame: int, method: str, top_k: int
) -> dict:
    """Rank joints for the warning that fired on `frame`."""
    window_index, position = locate_window(item, frame)
    window = item["windows"][window_index]
    relevance = joint_relevance(loaded.model, window, position, loaded.device, method=method)
    return {
        "method": method,
        "phrase": relevance.phrase(top_k),
        "top_joints": [
            {"joint": name, "index": int(JOINT_NAMES.index(name)), "relevance": round(score, 4)}
            for name, score in relevance.top_k(top_k)
        ],
        "relevance": [round(float(v), 5) for v in relevance.scores],
    }


def analyze(
    keypoints: np.ndarray,
    fps: float = 30.0,
    impact_frame: int | None = None,
    threshold: float | None = None,
    persistence: int | None = None,
    explain: bool = True,
    top_k: int = 3,
    clip_id: str = "upload",
) -> dict:
    """Score a keypoint sequence and describe every warning it produces.

    Args:
        keypoints: `[T, 17, 3]` of `(x, y, confidence)` in pixels.
        fps: frame rate, which sets the units of lead time.
        impact_frame: `t*` if known; lead time is only defined when it is.
        threshold: overrides the checkpoint's `tau`. The checkpoint's value was
            selected on validation; overriding it is a demo affordance, not a
            way to report a better number.
        persistence: overrides `k`.
        explain: attach joint evidence to each warning.
        top_k: how many joints to name.
        clip_id: identifier echoed back in the response.

    Returns:
        A JSON-ready dict of scores, warnings and timing.
    """
    started = time.perf_counter()
    loaded = load_model()

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"keypoints must be [T, {NUM_JOINTS}, 3] of (x, y, confidence); "
            f"got {list(keypoints.shape)}"
        )
    if keypoints.shape[0] > MAX_FRAMES:
        raise ValueError(
            f"{keypoints.shape[0]} frames exceeds the {MAX_FRAMES}-frame limit "
            f"(~{MAX_FRAMES / 30:.0f}s at 30fps); send a shorter segment"
        )
    if keypoints.shape[0] < loaded.features.window:
        raise ValueError(
            f"need at least {loaded.features.window} frames (the model's window); "
            f"got {keypoints.shape[0]}"
        )
    if impact_frame is not None and not 0 <= impact_frame < keypoints.shape[0]:
        raise ValueError(f"impact_frame {impact_frame} outside a {keypoints.shape[0]}-frame clip")

    clip = _as_clip(keypoints, fps, impact_frame, clip_id)
    dataset = ClipDataset([clip], loaded.features, loaded.labels)
    item = dataset[0]
    scored = score_clip(loaded.model, item, loaded.device, chunk_size=CHUNK_SIZE)

    decision = DecisionConfig(
        threshold=float(threshold if threshold is not None else loaded.decision.threshold),
        persistence=int(persistence if persistence is not None else loaded.decision.persistence),
        refractory_frames=loaded.decision.refractory_frames,
    )

    method = str(loaded.config.get("explain", {}).get("method", "attention"))
    w_pre = loaded.labels.w_pre
    warnings = []
    for frame in all_triggers(scored.scores, decision):
        entry = {
            "frame": int(frame),
            "time_s": round(frame / clip.fps, 3),
            "score": round(float(scored.scores[frame]), 4),
            "lead_time": None,
            "within_imminent_window": None,
        }
        if impact_frame is not None:
            lead = (impact_frame - frame) / clip.fps
            entry["lead_time"] = round(float(lead), 3) if frame <= impact_frame else None
            # A trigger before the imminent window is scored as a false alarm by
            # evaluation/metrics.py, so the demo must not present it as a hit.
            entry["within_imminent_window"] = bool(
                impact_frame - w_pre <= frame <= impact_frame
            )
        if explain:
            entry["evidence"] = _explain(loaded, item, frame, method, top_k)
        warnings.append(entry)

    # Copy out the one array the response needs, then hand the window stack and
    # every intermediate back before serialising. On a box where torch already
    # owns 80% of the memory, waiting for the collector's own schedule is what
    # turns a working request into a killed container on the next one.
    scores = [round(float(s), 5) for s in scored.scores]
    num_frames = int(clip.num_frames)
    del item, dataset, scored, clip
    gc.collect()

    return {
        "clip_id": clip_id,
        "frames": num_frames,
        "fps": float(fps),
        "impact_frame": impact_frame,
        "scores": scores,
        "warnings": warnings,
        "decision": {
            "threshold": decision.threshold,
            "persistence": decision.persistence,
            "refractory_frames": decision.refractory_frames,
        },
        "imminent_window": (
            [max(0, impact_frame - w_pre), impact_frame] if impact_frame is not None else None
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def faithfulness(
    keypoints: np.ndarray,
    frame: int,
    fps: float = 30.0,
    impact_frame: int | None = None,
    num_random: int = 3,
    baseline: str = "zero",
    clip_id: str = "upload",
) -> dict:
    """Run the deletion/insertion test on one warning (RQ2).

    Deliberately a separate endpoint. It costs `(1 + num_random) x 2 x 18`
    forward passes - a few seconds on a shared CPU - where scoring a whole clip
    costs one batch. Folding it into `/analyze` would make every request slow to
    produce a number most requests do not ask for.
    """
    started = time.perf_counter()
    loaded = load_model()

    keypoints = np.asarray(keypoints, dtype=np.float32)
    clip = _as_clip(keypoints, fps, impact_frame, clip_id)
    dataset = ClipDataset([clip], loaded.features, loaded.labels)
    item = dataset[0]

    window_index, position = locate_window(item, int(frame))
    window = item["windows"][window_index]
    method = str(loaded.config.get("explain", {}).get("method", "attention"))
    relevance = joint_relevance(loaded.model, window, position, loaded.device, method=method)

    curves = faithfulness_curves(
        model=loaded.model,
        window=window,
        frame=position,
        relevance=relevance.scores,
        device=loaded.device,
        baseline=baseline,
        num_random=max(1, min(int(num_random), 8)),
        seed=0,
        method=method,
    )

    return {
        "frame": int(frame),
        "method": method,
        "baseline": baseline,
        "num_random": num_random,
        "deletion": [round(float(v), 5) for v in curves.deletion],
        "insertion": [round(float(v), 5) for v in curves.insertion],
        "deletion_random": [round(float(v), 5) for v in curves.deletion_random],
        "insertion_random": [round(float(v), 5) for v in curves.insertion_random],
        "deletion_auc": round(curves.deletion_auc, 4),
        "insertion_auc": round(curves.insertion_auc, 4),
        "deletion_gap": round(curves.deletion_gap, 4),
        "insertion_gap": round(curves.insertion_gap, 4),
        # Both gaps are signed so that positive means "beats a random joint
        # ordering". Zero or negative is the honest negative result, not a bug.
        "faithful": bool(curves.deletion_gap > 0 and curves.insertion_gap > 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@lru_cache(maxsize=1)
def demo_clips() -> dict[str, ClipRecord]:
    """The bundled demo clips, keyed by id."""
    if not DEMO_CLIPS_DIR.exists():
        return {}
    return {clip.clip_id: clip for clip in load_cache(DEMO_CLIPS_DIR)}


def demo_clip_summaries() -> list[dict]:
    """Metadata for the clip picker."""
    return [
        {
            "clip_id": clip.clip_id,
            "label": clip.label,
            "activity": clip.activity,
            "num_frames": clip.num_frames,
            "fps": clip.fps,
            "duration_s": round(clip.duration_s, 2),
            "impact_frame": clip.impact_frame,
            "source": clip.source,
        }
        for clip in sorted(demo_clips().values(), key=lambda c: (c.label != "fall", c.clip_id))
    ]


def demo_clip_payload(clip_id: str) -> dict:
    """One demo clip's full keypoints, ready to send to the client."""
    clips = demo_clips()
    if clip_id not in clips:
        raise KeyError(clip_id)
    clip = clips[clip_id]
    return {
        "clip_id": clip.clip_id,
        "label": clip.label,
        "activity": clip.activity,
        "fps": clip.fps,
        "impact_frame": clip.impact_frame,
        "num_frames": clip.num_frames,
        "source": clip.source,
        "keypoints": np.round(clip.keypoints, 3).tolist(),
    }
