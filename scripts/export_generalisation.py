"""Export the held-out-domain evaluation for the website.

    python scripts/export_generalisation.py --data-root "D:/projects/Fall before impact"

Writes `frontend/lib/generalisation.json`: the four protocols with the measured
size of each split, and - once `results/xdomain/runs.csv` exists - the per-seed
results of every (protocol, arm) cell next to the in-domain reference for the same
arm. The page shows "pending" for any cell with no runs; nothing is estimated.

Importing the `data` package pulls in torch, which this script does not need, so
the package is stubbed out and only `data.clips` / `data.splits` are loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_pkg = types.ModuleType("data")
_pkg.__path__ = [str(ROOT / "data")]
sys.modules.setdefault("data", _pkg)
sys.path.insert(0, str(ROOT))

from data.clips import load_cache  # noqa: E402
from data.splits import filter_clips, holdout_split  # noqa: E402

SEEDS = (1, 2, 3)
ARMS = {
    "ctrl": {"label": "Control · (x, y, vx, vy)", "in_domain": "combined_xyv"},
    "phases": {"label": "Arm C · full features + four-phase head", "in_domain": "combined_phases"},
}
PROTOCOLS = [
    ("urfd_to_le2i", "UR Fall → Le2i", "UR Fall only", "Every Le2i clip", {"source": "le2i"}),
    ("le2i_to_urfd", "Le2i → UR Fall", "Le2i only", "Every UR Fall clip", {"source": "urfd"}),
    ("leave_coffee_room_out", "Leave Coffee room out", "UR Fall + Le2i Home", "Le2i Coffee room",
     {"source": "le2i", "scene": "coffee_room"}),
    ("leave_home_out", "Leave Home out", "UR Fall + Le2i Coffee room", "Le2i Home",
     {"source": "le2i", "scene": "home"}),
]


def stage_size(clips, split, stage):
    chosen = filter_clips(clips, split.subjects_for(stage))
    return {
        "clips": len(chosen),
        "falls": sum(c.is_fall for c in chosen),
        "normal_minutes": round(sum(c.num_frames / c.fps for c in chosen if not c.is_fall) / 60, 1),
    }


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def flt(row, key):
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if v != v else round(v, 4)


def cell(rows):
    per_seed = []
    for row in sorted(rows, key=lambda r: int(float(r["seed"]))):
        falls, warned = flt(row, "num_falls"), flt(row, "num_warned")
        per_seed.append({
            "seed": int(float(row["seed"])),
            "recall": round(warned / falls, 4) if falls else None,
            "false_alarms": flt(row, "num_false_alarms"),
            "normal_hours": flt(row, "negative_hours"),
            "lead": flt(row, "mean_lead_time"),
            "auc": flt(row, "frame_auc"),
        })
    return per_seed


def in_domain(rows, config):
    """Pooled recall and total false alarms over the confirmation grid's folds."""
    mine = [r for r in rows if r["config"] == config and int(float(r["seed"])) in SEEDS]
    falls = sum(flt(r, "num_falls") or 0 for r in mine)
    if not falls:
        return None
    return {
        "runs": len(mine),
        "recall": round(sum(flt(r, "num_warned") or 0 for r in mine) / falls, 4),
        "seeds": sorted({int(float(r["seed"])) for r in mine}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=str(ROOT),
                        help="tree holding data/cache and results/ (the main checkout)")
    parser.add_argument("--out", default=str(ROOT / "frontend" / "lib" / "generalisation.json"))
    args = parser.parse_args()
    root = Path(args.data_root)

    clips = load_cache(root / "data/cache/urfd") + load_cache(root / "data/cache/le2i")
    xd_rows = read_runs(root / "results/xdomain/runs.csv")
    confirm_rows = read_runs(root / "results/combined_confirm/runs.csv")

    protocols = []
    done = 0
    for pid, title, train_text, test_text, domain in PROTOCOLS:
        split = holdout_split(clips, domain, val_fraction=0.25, seed=0)
        arms = {}
        for aid, meta in ARMS.items():
            rows = [r for r in xd_rows if r["config"] == f"xd_{pid}_{aid}"
                    and int(float(r["seed"])) in SEEDS]
            done += len(rows)
            arms[aid] = {"label": meta["label"], "runs": cell(rows),
                         "in_domain": in_domain(confirm_rows, meta["in_domain"])}
        protocols.append({
            "id": pid, "title": title, "train": train_text, "test": test_text,
            "sizes": {s: stage_size(clips, split, s) for s in ("train", "val", "test")},
            "arms": arms,
        })

    expected = len(PROTOCOLS) * len(ARMS) * len(SEEDS)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": "results/analysis/PLAN_XDOMAIN.txt",
        "status": "pending" if done == 0 else "complete" if done >= expected else "partial",
        "runs_done": done, "runs_expected": expected,
        "limits": [
            "Both datasets are acted falls by young volunteers.",
            "Le2i has no actor labels: holding out a scene removes the room and camera, "
            "not necessarily the people.",
            "Held-out test sets contain only 4-9 minutes of normal activity, so one false "
            "alarm is roughly 7-15 per hour. Recall is the better-powered quantity.",
            "These runs describe generalisation; they cannot confirm the four-phase "
            "hypothesis, which PLAN.txt alone decides.",
        ],
        "protocols": protocols,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: status={payload['status']} ({done}/{expected} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
