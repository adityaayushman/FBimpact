"""Stages B-C: labelling, windowing and the leakage guards.

These are the tests that protect the validity of the headline number rather than
the correctness of a function. If `test_split_is_subject_disjoint` or
`test_causal_model_ignores_the_future` ever fails, every lead time the project
reports is meaningless, so they are worth more than their line count suggests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.labels import IGNORE_INDEX, LabelConfig, frame_labels, time_to_impact
from data.normalize import add_velocity, normalise_clip
from data.skeleton import NUM_JOINTS, flip_permutation
from data.splits import assert_subject_disjoint, single_split, subject_folds
from data.synthetic import make_clip, make_dataset
from data.windows import slice_windows, window_starts


def test_imminent_window_is_w_pre_frames_before_impact():
    labels = frame_labels(100, impact_frame=60, config=LabelConfig(w_pre=20))
    assert labels[39] == 0
    assert labels[40] == 1          # t* - w_pre
    assert labels[60] == 1          # t* itself
    assert (labels[40:61] == 1).all()


def test_post_impact_frames_are_ignored_by_default():
    labels = frame_labels(100, impact_frame=60, config=LabelConfig(w_pre=20))
    assert (labels[61:] == IGNORE_INDEX).all(), (
        "post-impact frames must be masked, or the task becomes post-fall detection"
    )


@pytest.mark.parametrize(
    "policy,expected", [("negative", 0), ("positive", 1), ("ignore", IGNORE_INDEX)]
)
def test_post_impact_policies(policy, expected):
    labels = frame_labels(80, 50, LabelConfig(w_pre=10, post_impact=policy))
    assert labels[70] == expected


def test_adl_clip_is_entirely_negative():
    labels = frame_labels(120, impact_frame=None)
    assert (labels == 0).all()
    assert np.isinf(time_to_impact(120, None)).all()


def test_time_to_impact_is_positive_before_and_negative_after():
    tti = time_to_impact(60, impact_frame=30, fps=30.0)
    assert tti[0] == pytest.approx(1.0)
    assert tti[30] == pytest.approx(0.0)
    assert tti[45] == pytest.approx(-0.5)


def test_windows_cover_the_end_of_the_clip():
    starts = window_starts(num_frames=100, window=30, stride=7)
    assert starts[0] == 0
    assert starts[-1] == 70, "the final frames, which contain t*, must be covered"
    assert len(set(starts)) == len(starts)


def test_short_clips_yield_no_windows():
    assert window_starts(num_frames=10, window=30, stride=1) == []


def test_slice_windows_preserves_alignment():
    features = np.arange(4 * 50 * NUM_JOINTS, dtype=np.float32).reshape(4, 50, NUM_JOINTS)
    labels = frame_labels(50, 40, LabelConfig(w_pre=10))
    tti = time_to_impact(50, 40)
    windows = slice_windows("c", features, labels, tti, window=20, stride=10)
    for w in windows:
        assert np.array_equal(w.features, features[:, w.start : w.start + 20, :])
        assert np.array_equal(w.labels, labels[w.start : w.start + 20])


def test_velocity_is_causal():
    """A centred difference would leak one frame of the future into every feature."""
    xy = np.zeros((10, NUM_JOINTS, 2), dtype=np.float32)
    xy[5:] = 1.0                                   # a step at frame 5
    features = add_velocity(xy, fps=1.0)
    velocity = features[2:]                        # [2, T, V]
    assert velocity[:, 4, :].max() == 0.0, "velocity moved before the position did"
    assert velocity[:, 5, :].max() == pytest.approx(1.0)


def test_normalisation_removes_translation_and_scale():
    rng = np.random.default_rng(0)
    clip = make_clip("c", "S00", "walk", rng)
    shifted = clip.keypoints.copy()
    shifted[..., :2] = shifted[..., :2] * 2.0 + np.array([300.0, -120.0], dtype=np.float32)

    a, _ = normalise_clip(clip.keypoints, fps=clip.fps)
    b, _ = normalise_clip(shifted, fps=clip.fps)
    assert np.allclose(a, b, atol=1e-3), (
        "a person twice as close to the camera must produce the same features"
    )


def test_low_confidence_joints_are_interpolated():
    rng = np.random.default_rng(1)
    clip = make_clip("c", "S00", "walk", rng)
    corrupted = clip.keypoints.copy()
    corrupted[10:15, 5, 2] = 0.0                   # drop a shoulder for five frames
    corrupted[10:15, 5, :2] = 9999.0               # with garbage coordinates

    features, valid = normalise_clip(corrupted, fps=clip.fps)
    assert not valid[10:15, 5].any()
    assert np.abs(features[:, 10:15, 5]).max() < 50.0, "garbage survived interpolation"


def test_flip_permutation_is_an_involution():
    perm = flip_permutation()
    assert np.array_equal(perm[perm], np.arange(NUM_JOINTS))


def test_split_is_subject_disjoint():
    clips = make_dataset(num_subjects=8, falls_per_subject=2, adls_per_subject=2)
    split = single_split(clips, seed=0)
    assert_subject_disjoint(split)                 # raises on leakage
    assert set(split.train) and set(split.val) and set(split.test)


def test_every_fold_is_subject_disjoint_and_covers_all_subjects():
    clips = make_dataset(num_subjects=10, falls_per_subject=1, adls_per_subject=1)
    folds = subject_folds(clips, num_folds=5, seed=0)
    covered: set[str] = set()
    for split in folds:
        assert_subject_disjoint(split)
        covered |= set(split.test)
    assert covered == {c.subject for c in clips}, "some subject is never tested on"


def test_causal_model_ignores_the_future():
    """Perturbing frame t+1 must not change the score at frame t.

    This is the property the whole lead-time claim rests on: if it fails, the
    model is reading the impact it claims to be predicting.
    """
    from models.stgcn import STGCN, STGCNConfig

    torch.manual_seed(0)
    model = STGCN(STGCNConfig(in_channels=4, blocks=((32, 1), (32, 2)), causal=True)).eval()
    x = torch.randn(1, 4, 30, NUM_JOINTS, 1)

    with torch.no_grad():
        before = model(x)
        perturbed = x.clone()
        perturbed[:, :, 20:, :, :] += 5.0
        after = model(perturbed)

    assert torch.allclose(before[:, :20], after[:, :20], atol=1e-5), (
        "changing future frames changed a past score: the model is not causal"
    )
    assert not torch.allclose(before[:, 20:], after[:, 20:], atol=1e-3), (
        "the perturbation had no effect at all; the test is not testing anything"
    )
