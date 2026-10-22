"""The held-out-domain split is the realistic-data protocol; it must not leak."""

from __future__ import annotations

import numpy as np
import pytest

from data.clips import ClipRecord
from data.splits import assert_subject_disjoint, filter_clips, holdout_split, scene_family


def clip(clip_id, source, view, subject=None, fall=False):
    return ClipRecord(
        clip_id=clip_id,
        subject=subject or clip_id,
        keypoints=np.zeros((12, 17, 3), dtype=np.float32),
        label="fall" if fall else "adl",
        impact_frame=8 if fall else None,
        source=source,
        view=view,
    )


@pytest.fixture
def clips():
    out = []
    for i in range(10):
        out.append(clip(f"urfd_{i}", "urfd", "default", fall=i < 4))
    for scene in ("coffee_room_01", "coffee_room_02", "home_01", "home_02"):
        for i in range(4):
            out.append(clip(f"le2i_{scene}_{i}", "le2i", scene, fall=i < 2))
    return out


def test_scene_family_strips_the_recording_index():
    assert scene_family("coffee_room_01") == "coffee_room"
    assert scene_family("home_02") == "home"
    assert scene_family("default") == "default"


def test_cross_dataset_test_side_is_exactly_the_held_out_source(clips):
    split = holdout_split(clips, {"source": "le2i"})
    test = filter_clips(clips, split.test)
    train = filter_clips(clips, split.train) + filter_clips(clips, split.val)
    assert test and all(c.source == "le2i" for c in test)
    assert train and all(c.source == "urfd" for c in train)
    assert_subject_disjoint(split)


def test_scene_holdout_removes_both_recordings_of_the_scene(clips):
    split = holdout_split(clips, {"source": "le2i", "scene": "coffee_room"})
    test = filter_clips(clips, split.test)
    rest = filter_clips(clips, split.train) + filter_clips(clips, split.val)
    assert {c.view for c in test} == {"coffee_room_01", "coffee_room_02"}
    assert not any(scene_family(c.view) == "coffee_room" and c.source == "le2i" for c in rest)
    # The other Le2i scene and UR Fall are still available for training.
    assert {"home_01", "home_02", "default"} <= {c.view for c in rest}


def test_validation_comes_from_the_training_side_only(clips):
    split = holdout_split(clips, {"source": "urfd"})
    assert set(split.val).isdisjoint(split.test)
    assert all(not s.startswith("urfd_") for s in split.val)


def test_deterministic_for_a_seed_and_different_across_seeds(clips):
    a = holdout_split(clips, {"source": "le2i"}, seed=3)
    b = holdout_split(clips, {"source": "le2i"}, seed=3)
    c = holdout_split(clips, {"source": "le2i"}, seed=4)
    assert a == b
    assert a.val != c.val or a.train != c.train


def test_a_subject_straddling_the_boundary_is_an_error_not_a_silent_fix(clips):
    clips.append(clip("le2i_extra", "le2i", "home_01", subject="urfd_0"))
    with pytest.raises(ValueError, match="not separable"):
        holdout_split(clips, {"source": "le2i"})


def test_an_unmatched_or_empty_domain_is_rejected(clips):
    with pytest.raises(ValueError, match="no clips match"):
        holdout_split(clips, {"source": "nonexistent"})
    with pytest.raises(ValueError, match="empty domain"):
        holdout_split(clips, {})
    with pytest.raises(ValueError, match="unknown domain keys"):
        holdout_split(clips, {"camera": "x"})


def test_build_split_wires_the_holdout_mode(clips):
    from data.build import build_split

    cfg = {"data": {"split": {"mode": "holdout", "test": {"source": "le2i"}}}}
    split = build_split(cfg, clips)
    assert set(split.test) == {c.subject for c in clips if c.source == "le2i"}
    with pytest.raises(ValueError, match="needs data.split.test"):
        build_split({"data": {"split": {"mode": "holdout"}}}, clips)
