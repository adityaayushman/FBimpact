"""False alarms at matched recall: decision-rule condition 3.

    python scripts/matched_recall.py

Why this exists. A lower false-alarm count at the default threshold can be nothing
more than a model whose scores are shifted lower: raise a threshold and false
alarms fall, but so does recall. The pre-registered rule therefore requires a
treatment to have fewer false alarms *at the same recall* before the site may say
it reduces false alarms in general (results/analysis/PLAN.txt, section 3.4).

Method, fixed in advance:
  1. For each (arm, seed), score every fold's held-out clips with that fold's own
     checkpoint, pool the streams, and sweep (tau, k) over the pooled set -
     exactly the procedure `scripts/export_curve.py` uses.
  2. Take the Pareto frontier of (false alarms/hour, recall).
  3. At each pre-fixed recall level, read off false alarms/hour by linear
     interpolation between the two frontier points that straddle the level.
     If the curve never reaches the level, report "not reached" - never
     extrapolate. If the lowest-false-alarm point already exceeds the level, the
     value is that point's rate and is flagged "at_floor".

These points come from a sweep over the test clips, so they are a robustness
check on the headline contrast and not a primary endpoint: a threshold chosen
after seeing test scores is exactly what the frozen-on-validation operating point
exists to avoid.

Torch is imported lazily so `fa_at_recall` can be unit-tested without it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEVELS = (0.60, 0.65)          # fixed before any confirmatory result was read


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pareto frontier of `(fa_per_hour, recall)`: no kept point is beaten on both.

    Sorted by false-alarm rate ascending; a point survives only if its recall
    strictly exceeds every cheaper point's.
    """
    best, out = -1.0, []
    for fa, recall in sorted(points, key=lambda p: (p[0], -p[1])):
        if recall > best:
            out.append((fa, recall))
            best = recall
    return out


def fa_at_recall(points: list[tuple[float, float]], level: float) -> tuple[float | None, str]:
    """False-alarm rate at `level` recall, with a status string.

    Returns `(value, status)` where status is "ok", "at_floor" (the cheapest
    frontier point already meets the level, so the true value may be lower than
    any observed) or "not_reached".
    """
    front = frontier(points)
    if not front or front[-1][1] < level:
        return None, "not_reached"
    if front[0][1] >= level:
        return front[0][0], "at_floor"
    for (fa0, r0), (fa1, r1) in zip(front, front[1:]):
        if r0 < level <= r1:
            w = (level - r0) / (r1 - r0)
            return fa0 + w * (fa1 - fa0), "ok"
    return front[-1][0], "ok"            # unreachable given the checks above


def main(argv: list[str] | None = None) -> int:
    import torch

    from scripts.export_curve import pooled_scores, sweep   # noqa: E402  (needs torch)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", nargs="*", default=["results/combined", "results/combined_confirm"])
    parser.add_argument("--arms", nargs="*",
                        default=["combined_xyv", "combined", "combined_phases", "combined_nocons"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="results/analysis/matched_recall.csv")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index: dict[tuple[str, int], list[Path]] = {}
    for directory in args.results:
        path = ROOT / directory / "runs.csv"
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            rd = Path(row["run_dir"])
            index.setdefault((row["config"], int(float(row["seed"]))), []).append(
                rd if rd.is_absolute() else ROOT / rd)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            dirs = index.get((arm, seed), [])
            if len(dirs) != 5:
                continue                                  # only complete 5-fold cells
            items, w_pre = pooled_scores(sorted(dirs), device)
            curve = sweep(items, w_pre)
            points = [(p["false_alarms_per_hour"], p["recall"]) for p in curve]
            for level in LEVELS:
                value, status = fa_at_recall(points, level)
                rows.append({"arm": arm, "seed": seed, "recall_level": level,
                             "fa_per_hour": "" if value is None else round(value, 2), "status": status})
            print(f"  {arm} seed {seed}: "
                  + ", ".join(f"@{l:.2f} -> {fa_at_recall(points, l)[0]}" for l in LEVELS))
            del items, curve

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "seed", "recall_level", "fa_per_hour", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
