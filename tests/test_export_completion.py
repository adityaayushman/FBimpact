from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_completion as ec  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_status_is_pending_partial_or_complete():
    assert ec.status(0, 20) == "pending"
    assert ec.status(7, 20) == "partial"
    assert ec.status(20, 20) == "complete"
    assert ec.status(25, 20) == "complete"


def test_nothing_run_yet_exports_pending_and_no_fabricated_numbers(tmp_path):
    out = ec.build(tmp_path)
    for key in ("rq3", "ablation_urfd", "ablation_le2i"):
        assert out[key]["status"] == "pending"
        assert out[key]["arms"] == [] and out[key]["contrasts"] == []


def test_only_the_planned_configs_and_seeds_are_counted(tmp_path):
    rows = ([{"config": "combined_preimpact", "seed": s, "fold": f} for s in (1, 2) for f in range(5)]
            + [{"config": "combined", "seed": 1, "fold": 0},               # arm B is part of E3 now
               {"config": "combined_preimpact", "seed": 0, "fold": 0},     # seed 0 is not part of E3
               {"config": "combined_xyv", "seed": 1, "fold": 0}])          # a different arm
    write(tmp_path / "results/combined_rq3/runs.csv", rows)
    out = ec.build(tmp_path)
    assert out["rq3"]["runs"] == 11 and out["rq3"]["status"] == "partial"


def test_verdict_text_is_passed_through_unchanged(tmp_path):
    write(tmp_path / "results/analysis/rq3/contrasts.csv", [{
        "contrast": "G-B", "means": "pre-impact loss", "d_fa": "3.5", "d_recall": "-0.02",
        "fa_verdict": "not shown", "recall_verdict": "not shown", "fa_two_sided": "not shown"}])
    c = ec.build(tmp_path)["rq3"]["contrasts"][0]
    assert c["fa_verdict"] == "not shown" and c["recall_verdict"] == "not shown"
    assert c["d_fa"] == 3.5 and c["d_recall"] == -0.02
