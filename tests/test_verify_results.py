"""The results verifier must be able to fail: a check that never fails proves nothing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_results as vr  # noqa: E402

HAVE_RUNS = (ROOT / "results" / "urfd" / "runs.csv").exists() and (ROOT / "results" / "combined_confirm" / "runs.csv").exists()
needs_runs = pytest.mark.skipif(not HAVE_RUNS, reason="run folders are not committed; this checks them where they exist")


def test_recall_is_pooled_by_summing_before_dividing():
    rows = [{"num_warned": "8", "num_falls": "8"}, {"num_warned": "0", "num_falls": "2"}]
    assert vr.pooled_recall(rows) == 0.8                      # not the mean of 1.0 and 0.0


def test_a_diff_that_only_adds_lines_removes_nothing():
    diff = "--- a/PLAN.txt\n+++ b/PLAN.txt\n@@ -1,2 +1,4 @@\n unchanged\n+ADDENDUM 2026-10-22\n+more\n"
    assert vr.removed_lines(diff) == []


def test_a_diff_that_rewrites_a_line_is_caught():
    diff = "--- a/PLAN.txt\n+++ b/PLAN.txt\n@@ -1,2 +1,2 @@\n-Seeds: 1, 2, 3, 4\n+Seeds: 1, 2\n"
    assert vr.removed_lines(diff) == ["-Seeds: 1, 2, 3, 4"]


@needs_runs
def test_a_wrong_published_recall_fails_the_check(monkeypatch):
    monkeypatch.setattr(vr, "README_UR_FALL", {"ours_preimpact": (0.700, 15)})
    ledger = vr.Ledger()
    vr.check_ur_fall(ledger)
    assert ledger.failed == 1


@needs_runs
def test_a_wrong_run_count_fails_the_check(monkeypatch):
    monkeypatch.setattr(vr, "README_UR_FALL", {"ours_preimpact": (0.633, 14)})
    ledger = vr.Ledger()
    vr.check_ur_fall(ledger)
    assert ledger.failed == 1


@needs_runs
def test_the_published_figures_pass():
    ledger = vr.Ledger()
    vr.check_ur_fall(ledger)
    vr.check_confirmation(ledger)
    assert ledger.failed == 0 and len(ledger.items) == 11
