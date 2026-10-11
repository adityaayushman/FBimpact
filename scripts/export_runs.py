"""Export every individual run, for the website's experiment explorer.

    python scripts/export_runs.py

`results.json` holds per-arm aggregates, which is what a headline needs but hides
what a sceptic most wants to see: the individual runs, and how far apart they land
for the same configuration. This writes one record per (arm, seed, fold) to
`frontend/lib/runs.json` so the explorer can show the seed-to-seed spread directly.

Reads only the `runs.csv` files that `scripts/run_ablations.py` appends to; nothing
here is computed from anything but those rows. No torch.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_results as ex  # noqa: E402  (label table; numpy/csv only)

BENCHMARKS = [
    ("combined_confirm", "Combined · confirmation seeds", "combined_confirm", True),
    ("combined", "Combined · seed 0", "combined", False),
    ("le2i", "Le2i", "le2i", False),
    ("urfd", "UR Fall", "urfd", False),
    ("synthetic", "Synthetic fixture", "bench", False),
]

LABELS = {stem: label for stem, label, _ in ex.VARIANTS} | {
    "combined_nocons": "D · phases, no consistency",
}


def num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, 5)


def main() -> int:
    runs = []
    for bench_id, bench_name, directory, confirmatory in BENCHMARKS:
        path = ROOT / "results" / directory / "runs.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seed = int(float(row["seed"]))
                runs.append({
                    "benchmark": bench_id, "benchmark_name": bench_name,
                    "arm": row["config"], "arm_label": LABELS.get(row["config"], row["config"]),
                    "seed": seed, "fold": int(float(row.get("fold") or 0)),
                    # Only seeds 1-4 of the confirmation grid count as evidence;
                    # seed 0 generated the hypothesis. Everything else is simply
                    # a historical run and says nothing about confirmation.
                    "role": ("confirmatory" if confirmatory and seed in (1, 2, 3, 4)
                             else "descriptive"),
                    "recall": num(row.get("recall")),
                    "lead": num(row.get("mean_lead_time")),
                    "false_alarms": num(row.get("num_false_alarms")),
                    "fa_per_hour": num(row.get("false_alarms_per_hour")),
                    "warned": num(row.get("num_warned")), "falls": num(row.get("num_falls")),
                    "auc": num(row.get("frame_auc")),
                })

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "runs": runs}
    out = ROOT / "frontend" / "lib" / "runs.json"
    out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    by = {}
    for r in runs:
        by[r["benchmark"]] = by.get(r["benchmark"], 0) + 1
    print(f"wrote {out}: {len(runs)} runs  " + ", ".join(f"{k}={v}" for k, v in by.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
