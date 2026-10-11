"""The analysis script is the thing that decides what the site may claim, so its
arithmetic is tested directly. It imports no torch, so this file is safe to run
while a training job holds the machine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_arms as aa  # noqa: E402


def geom(frames=300, fps=30.0, impact=None, dataset="urfd", is_fall=False):
    return {"frames": frames, "fps": fps, "impact": impact, "dataset": dataset, "is_fall": is_fall}


def test_adl_negative_time_is_the_whole_clip():
    assert aa.negative_hours(geom(frames=3600, fps=30.0), w_pre=17) == pytest.approx(120 / 3600)


def test_fall_negative_time_stops_where_the_imminent_window_starts():
    """Frames inside the window and after impact are not 'normal' footage."""
    g = geom(frames=300, fps=30.0, impact=200, is_fall=True)
    assert aa.negative_hours(g, w_pre=17) == pytest.approx((200 - 17) / 30.0 / 3600.0)


def test_fall_with_impact_inside_the_window_has_no_negative_time():
    g = geom(frames=300, impact=10, is_fall=True)
    assert aa.negative_hours(g, w_pre=17) == 0.0


def test_negative_time_scales_with_fps():
    """25 fps and 30 fps clips of equal frame count are different durations."""
    a = aa.negative_hours(geom(frames=900, fps=30.0), 17)
    b = aa.negative_hours(geom(frames=900, fps=25.0), 17)
    assert b == pytest.approx(a * 30.0 / 25.0)


def rec(fall, warned=False, fa=0, dataset="urfd", neg_h=0.01, lead=None):
    return {"is_fall": fall, "warned": warned, "lead": lead, "fa": fa, "neg_h": neg_h, "dataset": dataset}


def test_summary_counts_recall_over_falls_only():
    records = {
        "f1": rec(True, True, lead=0.4), "f2": rec(True, False), "f3": rec(True, True, lead=0.6),
        "a1": rec(False, fa=2), "a2": rec(False, fa=1),
    }
    s = aa.summarise(records)
    assert s["falls"] == 3 and s["warned"] == 2
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["lead"] == pytest.approx(0.5)
    assert s["fa"] == 3


def test_lead_time_ignores_missed_falls():
    s = aa.summarise({"f1": rec(True, False), "f2": rec(True, True, lead=0.3)})
    assert s["lead"] == pytest.approx(0.3)


def make_runs(n_clips, treat_fa, ref_fa, seeds=(1, 2, 3, 4)):
    """Synthetic paired runs: every ADL clip gets the given per-clip alarm counts."""
    runs = {}
    for s in seeds:
        for arm, fa in (("T", treat_fa), ("R", ref_fa)):
            runs[(arm, s)] = {
                f"c{i}": rec(False, fa=fa(s, i), dataset="urfd" if i % 2 else "le2i")
                for i in range(n_clips)
            }
    return runs


def test_bootstrap_detects_a_real_consistent_reduction():
    runs = make_runs(80, treat_fa=lambda s, i: 0, ref_fa=lambda s, i: 1)
    universe = aa.ordered_clips(runs, ["T", "R"], [1, 2, 3, 4])
    out = aa.paired_bootstrap(runs, "T", "R", [1, 2, 3, 4], universe, n_boot=300)
    assert out["d_fa"] == pytest.approx(-80.0)
    assert out["d_fa_hi"] < 0


def test_bootstrap_does_not_invent_an_effect_from_noise():
    """Identical arms must give an interval that straddles zero."""
    rng = np.random.default_rng(0)
    table = {(s, i): int(rng.integers(0, 3)) for s in (1, 2, 3, 4) for i in range(80)}
    runs = make_runs(80, treat_fa=lambda s, i: table[(s, i)], ref_fa=lambda s, i: table[(s, i)])
    universe = aa.ordered_clips(runs, ["T", "R"], [1, 2, 3, 4])
    out = aa.paired_bootstrap(runs, "T", "R", [1, 2, 3, 4], universe, n_boot=300)
    assert out["d_fa"] == 0.0
    assert out["d_fa_lo"] <= 0.0 <= out["d_fa_hi"]


def test_bootstrap_reuses_one_clip_resample_for_every_seed():
    """If seeds were resampled independently the interval would be too narrow.

    With a treatment effect present in only ONE clip, per-seed independent
    resampling would average the clip's presence away across seeds; a shared
    resample keeps it, so the interval must be wide relative to the point.
    """
    runs = make_runs(40, treat_fa=lambda s, i: 0, ref_fa=lambda s, i: 4 if i == 0 else 0)
    universe = aa.ordered_clips(runs, ["T", "R"], [1, 2, 3, 4])
    out = aa.paired_bootstrap(runs, "T", "R", [1, 2, 3, 4], universe, n_boot=600)
    assert out["d_fa"] == pytest.approx(-4.0)
    # A shared resample draws the effective clip 0 times in a sizeable fraction of
    # resamples, so the upper bound reaches zero.
    assert out["d_fa_hi"] >= -1e-9


def test_universe_is_only_clips_present_in_every_cell():
    runs = make_runs(10, lambda s, i: 0, lambda s, i: 0)
    del runs[("R", 3)]["c4"]
    assert "c4" not in aa.ordered_clips(runs, ["T", "R"], [1, 2, 3, 4])
