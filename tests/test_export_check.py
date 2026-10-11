"""`export_results.py --check`: the gate between run directories and the website.

No torch is imported, so this runs alongside a training job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_results as ex  # noqa: E402


def row(variant, run_dirs, recall=0.7, fa=100.0):
    return {"variant": variant, "label": variant, "role": "ours", "runs": len(run_dirs),
            "recall": {"mean": recall, "std": 0.0},
            "false_alarms_per_hour": {"mean": fa, "std": 0.0},
            "run_dirs": run_dirs}


def payload(rows):
    return {"benchmarks": [{"id": "combined", "rows": rows}]}


@pytest.fixture
def root(tmp_path):
    for name in ("a", "b", "c"):
        d = tmp_path / "results" / name
        d.mkdir(parents=True)
        (d / "report_test.json").write_text("{}", encoding="utf-8")
    return tmp_path


def good_rows():
    return [row("combined_xyv", ["results/a"]), row("combined", ["results/b"]),
            row("combined_phases", ["results/c"])]


def test_a_complete_payload_passes(root):
    assert ex.check_payload(payload(good_rows()), root) == []


def test_a_missing_required_arm_is_reported(root):
    rows = [r for r in good_rows() if r["variant"] != "combined_phases"]
    problems = ex.check_payload(payload(rows), root)
    assert any("combined_phases" in p and "no runs" in p for p in problems)


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_a_missing_or_nan_headline_number_is_reported(root, bad):
    rows = good_rows()
    rows[1]["recall"]["mean"] = bad
    assert any("'recall'" in p for p in ex.check_payload(payload(rows), root))


def test_a_row_whose_source_directory_vanished_is_reported(root):
    rows = good_rows()
    rows[0]["run_dirs"] = ["results/gone"]
    assert any("results/gone" in p for p in ex.check_payload(payload(rows), root))


def test_a_row_with_no_recorded_sources_is_reported(root):
    rows = good_rows()
    rows[2]["run_dirs"] = []
    assert any("no source run directories" in p for p in ex.check_payload(payload(rows), root))


def test_benchmarks_without_required_arms_are_not_held_to_the_combined_list(root):
    other = {"benchmarks": [{"id": "synthetic", "rows": [row("baseline_stgcn", ["results/a"])]}]}
    assert ex.check_payload(other, root) == []
