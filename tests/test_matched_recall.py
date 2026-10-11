"""Matched-recall interpolation. Pure NumPy-free logic, so it runs beside training."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import matched_recall as mr  # noqa: E402


def test_frontier_drops_points_beaten_on_both_axes():
    pts = [(10.0, 0.5), (20.0, 0.4), (30.0, 0.7), (25.0, 0.7), (50.0, 0.9)]
    assert mr.frontier(pts) == [(10.0, 0.5), (25.0, 0.7), (50.0, 0.9)]


def test_interpolates_linearly_between_straddling_frontier_points():
    pts = [(10.0, 0.50), (30.0, 0.70)]
    value, status = mr.fa_at_recall(pts, 0.60)
    assert status == "ok" and value == pytest.approx(20.0)


def test_level_exactly_on_a_point_returns_that_point():
    pts = [(10.0, 0.50), (30.0, 0.70)]
    value, status = mr.fa_at_recall(pts, 0.70)
    assert status == "ok" and value == pytest.approx(30.0)


def test_unreached_level_is_reported_never_extrapolated():
    value, status = mr.fa_at_recall([(10.0, 0.40), (30.0, 0.55)], 0.60)
    assert value is None and status == "not_reached"


def test_cheapest_point_already_above_the_level_is_flagged_at_floor():
    """The true rate at that recall may be lower than anything observed."""
    value, status = mr.fa_at_recall([(5.0, 0.80), (30.0, 0.90)], 0.60)
    assert status == "at_floor" and value == pytest.approx(5.0)


def test_empty_curve_is_not_reached():
    assert mr.fa_at_recall([], 0.6) == (None, "not_reached")


def test_a_dominated_noisy_point_does_not_distort_the_interpolation():
    """A worse point between two frontier points must be ignored, not averaged in."""
    clean = mr.fa_at_recall([(10.0, 0.50), (30.0, 0.70)], 0.60)[0]
    noisy = mr.fa_at_recall([(10.0, 0.50), (20.0, 0.45), (30.0, 0.70)], 0.60)[0]
    assert noisy == pytest.approx(clean)
