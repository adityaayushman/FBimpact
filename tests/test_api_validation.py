"""Backend input validation: one test per documented outcome.

Imports numpy only, so it runs alongside a training job. The boundary cases for
dropout - at, just under and just over each limit - are the point: a limit that
is only tested away from its edge is a limit nobody knows the behaviour of.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.validation import (  # noqa: E402
    MAX_ABS_COORD,
    MAX_GAP_FRAMES,
    MAX_MISSING_FRACTION,
    ApiError,
    validate,
)

WINDOW, CAP = 30, 600


def clip(frames=120, conf=0.9):
    """A clean synthetic clip: every joint observed on every frame."""
    kp = np.zeros((frames, 17, 3))
    kp[..., 0] = np.linspace(100, 140, 17)[None, :] + np.arange(frames)[:, None] * 0.1
    kp[..., 1] = np.linspace(50, 300, 17)[None, :]
    kp[..., 2] = conf
    return kp.tolist()


def blank_frames(raw, frames):
    """Set the given frames entirely to null."""
    for f in frames:
        raw[f] = [[None, None, None] for _ in range(17)]
    return raw


def check(raw, fps=30.0, **kw):
    return validate(raw, fps, kw.get("min_frames", WINDOW), kw.get("max_frames", CAP), kw.get("impact"))


def code_of(raw, **kw):
    with pytest.raises(ApiError) as info:
        check(raw, **kw)
    return info.value.status, info.value.code


# -- structure --------------------------------------------------------------

def test_a_clean_clip_passes_untouched():
    out = check(clip())
    assert not out.degraded and not out.no_person and out.interpolated_frames == []
    assert out.keypoints.shape == (120, 17, 3)


def test_wrong_joint_count_is_bad_shape():
    assert code_of(np.zeros((120, 16, 3)).tolist()) == (422, "bad_shape")


def test_wrong_channel_count_is_bad_shape():
    assert code_of(np.zeros((120, 17, 2)).tolist()) == (422, "bad_shape")


def test_ragged_frames_are_bad_shape_not_a_crash():
    raw = clip()
    raw[5] = raw[5][:10]
    assert code_of(raw) == (422, "bad_shape")


def test_not_a_list_is_bad_shape():
    assert code_of({"a": 1}) == (422, "bad_shape")
    assert code_of("keypoints") == (422, "bad_shape")


def test_strings_inside_are_non_numeric():
    raw = clip()
    raw[3][2][0] = "abc"
    assert code_of(raw) == (422, "non_numeric")


def test_empty_clip():
    assert code_of([]) == (422, "empty_clip")


def test_too_short_states_the_minimum_and_never_pads():
    with pytest.raises(ApiError) as info:
        check(clip(WINDOW - 1))
    assert (info.value.status, info.value.code) == (422, "too_short")
    assert info.value.extra["minimum"] == WINDOW


def test_exactly_the_window_is_accepted():
    assert check(clip(WINDOW)).keypoints.shape[0] == WINDOW


def test_too_long_is_413_and_states_the_cap():
    with pytest.raises(ApiError) as info:
        check(clip(CAP + 1))
    assert (info.value.status, info.value.code) == (413, "too_long")
    assert info.value.extra["maximum"] == CAP


def test_exactly_the_cap_is_accepted():
    assert check(clip(CAP)).keypoints.shape[0] == CAP


@pytest.mark.parametrize("fps", [0.0, 4.9, 120.1, 1e9, float("nan"), float("inf")])
def test_fps_outside_the_range_is_bad_fps(fps):
    assert code_of(clip(), fps=fps) == (422, "bad_fps")


@pytest.mark.parametrize("fps", [5.0, 25.0, 30.0, 120.0])
def test_fps_inside_the_range_is_accepted(fps):
    check(clip(), fps=fps)


def test_impact_frame_outside_the_clip_is_rejected():
    assert code_of(clip(120), impact=120) == (422, "bad_impact_frame")
    assert code_of(clip(120), impact=-1) == (422, "bad_impact_frame")
    check(clip(120), impact=119)


# -- missing data -----------------------------------------------------------

def test_every_frame_missing_is_no_person_not_an_error_and_not_a_low_score():
    raw = blank_frames(clip(60), range(60))
    out = check(raw)
    assert out.no_person
    assert not np.any(out.keypoints)


def test_an_all_zero_clip_is_also_no_person():
    assert check(np.zeros((60, 17, 3)).tolist()).no_person


def test_a_few_null_frames_are_repaired_flagged_and_listed():
    """The case that matters: a brief detection loss must not be a 422."""
    raw = blank_frames(clip(120), range(70, 76))
    out = check(raw)
    assert out.degraded
    assert out.interpolated_frames == [70, 71, 72, 73, 74, 75]
    assert not np.isnan(out.keypoints).any()
    assert np.all(out.keypoints[70:76] == 0.0)   # zero conf: normalise() bridges it


def test_nan_and_infinity_tokens_count_as_missing_like_null():
    raw = clip(120)
    for j in range(17):
        raw[40][j] = [float("nan"), float("nan"), float("nan")]
        raw[41][j] = [float("inf"), float("-inf"), 0.9]
    out = check(raw)
    assert out.interpolated_frames == [40, 41]
    assert np.isfinite(out.keypoints).all()


def test_a_single_dropped_joint_is_not_a_missing_frame():
    """Partial occlusion is for the model's own interpolation, not the gate."""
    raw = clip(120)
    raw[50][5] = [None, None, None]
    out = check(raw)
    assert not out.degraded and out.interpolated_frames == []
    assert np.isfinite(out.keypoints).all()


def test_zero_confidence_frames_are_missing():
    raw = clip(120)
    for j in range(17):
        raw[60][j][2] = 0.0
    assert check(raw).interpolated_frames == [60]


# -- the dropout limits, at and either side of each edge --------------------

def test_gap_of_exactly_the_limit_is_repaired():
    out = check(blank_frames(clip(300), range(100, 100 + MAX_GAP_FRAMES)))
    assert out.degraded and len(out.interpolated_frames) == MAX_GAP_FRAMES


def test_gap_one_over_the_limit_is_refused_with_the_numbers():
    with pytest.raises(ApiError) as info:
        check(blank_frames(clip(300), range(100, 100 + MAX_GAP_FRAMES + 1)))
    assert (info.value.status, info.value.code) == (422, "too_much_missing")
    assert info.value.extra["longest_gap"] == MAX_GAP_FRAMES + 1


def test_gap_one_under_the_limit_is_repaired():
    out = check(blank_frames(clip(300), range(100, 100 + MAX_GAP_FRAMES - 1)))
    assert len(out.interpolated_frames) == MAX_GAP_FRAMES - 1


def scattered(frames, count):
    """`count` isolated missing frames, none adjacent, so only the fraction can fail."""
    step = frames // count
    return [i * step + 1 for i in range(count)]


def test_fraction_exactly_at_the_limit_is_repaired():
    frames = 200
    n = int(frames * MAX_MISSING_FRACTION)           # 40 frames = exactly 20%
    out = check(blank_frames(clip(frames), scattered(frames, n)))
    assert len(out.interpolated_frames) == n


def test_fraction_one_frame_over_the_limit_is_refused():
    frames = 200
    n = int(frames * MAX_MISSING_FRACTION) + 1
    with pytest.raises(ApiError) as info:
        check(blank_frames(clip(frames), scattered(frames, n)))
    assert info.value.code == "too_much_missing"
    assert info.value.extra["missing_fraction"] > MAX_MISSING_FRACTION


def test_fraction_one_frame_under_the_limit_is_repaired():
    frames = 200
    n = int(frames * MAX_MISSING_FRACTION) - 1
    assert check(blank_frames(clip(frames), scattered(frames, n))).degraded


# -- range ------------------------------------------------------------------

def test_non_pixel_coordinates_are_refused():
    raw = clip()
    raw[10][3][0] = MAX_ABS_COORD * 10
    assert code_of(raw) == (422, "out_of_range")


def test_large_but_legitimate_pixel_coordinates_are_accepted():
    raw = clip()
    raw[10][3][0] = 7680.0                            # 8K frame width
    check(raw)


def test_confidence_outside_unit_interval_is_clipped_and_noted():
    raw = clip()
    raw[5][2][2] = 7.0
    raw[6][2][2] = -3.0
    out = check(raw)
    assert out.clipped_confidence
    assert out.keypoints[..., 2].min() >= 0.0 and out.keypoints[..., 2].max() <= 1.0


def test_a_missing_frame_with_absurd_coordinates_is_not_range_checked():
    """Range applies to frames that remain observed, not to ones being discarded."""
    raw = clip(120)
    for j in range(17):
        raw[30][j] = [1e12, 1e12, 0.0]               # unusable: zero confidence
    out = check(raw)
    assert out.interpolated_frames == [30]


# -- the contract every error shares ---------------------------------------

def test_every_error_has_a_stable_body_shape():
    with pytest.raises(ApiError) as info:
        check([])
    body = info.value.body("req-123")
    assert set(body) == {"error"}
    assert {"code", "message", "request_id"} <= set(body["error"])
    assert body["error"]["request_id"] == "req-123"
