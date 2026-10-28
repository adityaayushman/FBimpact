"""Export the PLAN_COMPLETION.txt results for the website.

    python scripts/export_completion.py

Reads what `analyze_arms.py --mode rq3 | ablation-urfd | ablation-le2i` wrote and the run
counts, and writes `frontend/lib/completion.json`. Each experiment is "pending" (no complete
cell yet), "partial" (some) or "complete" (every planned run). Verdict wording is passed
through from the analysis, never rewritten here, so the page cannot promote "not shown".
No torch.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["ours_preimpact", "baseline_stgcn", "no_preimpact_loss", "no_temporal", "no_velocity", "no_grounding"]

EXPERIMENTS = {
    "rq3": {"runs_dir": "results/combined_rq3", "analysis": "results/analysis/rq3", "expected": 40, "configs": ["combined", "combined_preimpact"]},
    "ablation_urfd": {"runs_dir": "results/urfd_colab", "analysis": "results/analysis/ablation_urfd", "expected": 90, "configs": VARIANTS},
    "ablation_le2i": {"runs_dir": "results/le2i_colab", "analysis": "results/analysis/ablation_le2i", "expected": 90, "configs": VARIANTS},
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value if value not in ("", None) else None
    return None if v != v else round(v, 4)


def clean(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({k: (num(v) if k not in ("arm", "contrast", "means", "fa_verdict", "recall_verdict", "fa_two_sided",
                                              "seeds_fewer_fa", "seeds_lower_recall", "seeds_longer_lead", "cond3_matched_recall")
                        else v) for k, v in r.items()})
    return out


def count_runs(runs_csv: Path, configs: list[str], seeds: tuple[int, ...] | None) -> int:
    n = 0
    for r in read_csv(runs_csv):
        if r["config"] in configs and (seeds is None or int(float(r["seed"])) in seeds):
            n += 1
    return n


def status(done: int, expected: int) -> str:
    return "complete" if done >= expected else "partial" if done > 0 else "pending"


def build(root: Path = ROOT) -> dict:
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, spec in EXPERIMENTS.items():
        seeds = (1, 2, 3, 4) if key == "rq3" else (0, 1, 2)
        done = count_runs(root / spec["runs_dir"] / "runs.csv", spec["configs"], seeds)
        analysis = root / spec["analysis"]
        payload[key] = {
            "status": status(done, spec["expected"]),
            "runs": min(done, spec["expected"]), "expected": spec["expected"],
            "arms": clean(read_csv(analysis / "arms_summary.csv")),
            "contrasts": clean(read_csv(analysis / "contrasts.csv")),
        }
    return payload


def main() -> int:
    payload = build()
    out = ROOT / "frontend" / "lib" / "completion.json"
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    for key in EXPERIMENTS:
        p = payload[key]
        print(f"  {key:14s} {p['status']:9s} {p['runs']}/{p['expected']} runs, {len(p['arms'])} arms, {len(p['contrasts'])} contrasts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
