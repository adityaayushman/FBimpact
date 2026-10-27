"""The pooled faithfulness statistics must weight, pair and decide the way PLAN_COMPLETION.txt says."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_faithfulness as af  # noqa: E402


def test_pooled_weights_runs_by_their_explanation_count():
    est = af.pooled([0.0, 1.0], [1, 3])["mean"]
    assert abs(est - 0.75) < 1e-9


def test_a_single_run_has_no_interval_rather_than_a_fake_one():
    p = af.pooled([0.2], [5])
    assert p["lo"] is None and p["hi"] is None and p["runs"] == 1


def test_interval_is_deterministic_and_brackets_the_estimate():
    values, weights = list(np.linspace(-0.1, 0.3, 12)), [10] * 12
    a, b = af.pooled(values, weights), af.pooled(values, weights)
    assert a == b and a["lo"] <= a["mean"] <= a["hi"]


def _rows(method_gaps):
    rows = []
    for i in range(8):
        for method, gap in method_gaps.items():
            rows.append({"run": f"combined_seed{i}_x", "arm": "combined", "method": method, "n": 10,
                         "deletion_gap": gap + 0.01 * (i % 3), "insertion_gap": gap + 0.01 * (i % 2)})
    return rows


def test_a_clearly_better_method_is_reported_as_more_faithful_than_attention():
    out = af.compare_methods(_rows({"attention": 0.05, "occlusion": 0.20}))
    occ = next(e for e in out if e["arm"] == "combined" and e["method"] == "occlusion")
    assert occ["more_faithful_than_reference"] is True and occ["faithful"] is True


def test_identical_methods_are_not_called_better():
    out = af.compare_methods(_rows({"attention": 0.05, "gradient": 0.05}))
    grad = next(e for e in out if e["arm"] == "combined" and e["method"] == "gradient")
    assert grad["more_faithful_than_reference"] is False


def test_a_method_with_gaps_around_zero_is_not_called_faithful():
    rows = _rows({"attention": 0.0})
    for r in rows:
        r["deletion_gap"] = [-0.1, 0.1][hash(r["run"]) % 2]
    out = af.compare_methods(rows)
    att = next(e for e in out if e["arm"] == "combined" and e["method"] == "attention")
    assert att["faithful"] is False
