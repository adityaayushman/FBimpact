"""Pool the cross-validation folds into one operating-point curve.

    python scripts/export_curve.py --runs results/urfd --benchmark urfd

Section 10 asks for a curve rather than a single threshold, because a single
threshold is always open to the charge of having been chosen after seeing the
test set. This builds that curve honestly: each fold's checkpoint scores its
**own** held-out clips, the per-frame streams are pooled across folds so every
clip is represented exactly once, and the threshold sweep runs over the pooled
set. No clip is ever scored by a model that trained on it.

The result answers a question the summary table cannot: whether *any* operating
point reaches a tolerable false-alarm rate while keeping recall, or whether the
whole curve sits in unusable territory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import ClipDataset, FeatureConfig  # noqa: E402
from data.labels import LabelConfig  # noqa: E402
from data.splits import Split, filter_clips  # noqa: E402
from evaluation.decision import DecisionConfig  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402
from evaluation.runner import score_dataset  # noqa: E402


def pooled_scores(run_dirs: list[Path], device: torch.device):
    """Score every fold's held-out clips with that fold's own checkpoint."""
    from data.build import load_clips
    from eval import load_checkpoint

    pooled = []
    w_pre = None
    seen: set[str] = set()

    for run_dir in run_dirs:
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            continue
        model, cfg, blob = load_checkpoint(checkpoint, device)
        w_pre = int(cfg["labels"]["w_pre"])

        stored = blob.get("split") or {}
        if not stored.get("test"):
            print(f"  {run_dir.name}: no split recorded, skipped")
            continue

        clips = load_clips(cfg)
        subset = filter_clips(clips, Split(
            train=tuple(stored["train"]), val=tuple(stored["val"]),
            test=tuple(stored["test"]), fold=int(stored.get("fold", 0)),
        ).test)

        # Guard against a mis-specified sweep double-counting a clip, which
        # would quietly weight part of the data twice in the pooled curve.
        overlap = {c.clip_id for c in subset} & seen
        if overlap:
            raise SystemExit(f"clip(s) appear in two folds' test sets: {sorted(overlap)[:5]}")
        seen |= {c.clip_id for c in subset}

        dataset = ClipDataset(subset, FeatureConfig(**dict(cfg["features"])),
                              LabelConfig(**dict(cfg["labels"])))
        pooled.extend(score_dataset(model, dataset, device))
        print(f"  {run_dir.name}: {len(subset)} held-out clips")

    return pooled, w_pre


def sweep(items, w_pre: int, persistences=(1, 3, 5, 8)) -> list[dict]:
    """Recall / lead time / false alarms across thresholds, per persistence."""
    rows = []
    for k in persistences:
        for tau in np.round(np.arange(0.05, 1.0, 0.05), 2):
            report = evaluate(
                items,
                DecisionConfig(threshold=float(tau), persistence=int(k), refractory_frames=30),
                w_pre,
                "false_alarm",
            )
            rows.append({
                "threshold": float(tau),
                "persistence": int(k),
                "recall": round(report.recall, 4),
                "mean_lead_time": None if not np.isfinite(report.mean_lead_time)
                                  else round(report.mean_lead_time, 4),
                "false_alarms": int(report.num_false_alarms),
                "false_alarms_per_hour": round(report.false_alarms_per_hour, 2),
                "specificity": round(report.specificity, 4),
                "num_falls": int(report.num_falls),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="results/urfd")
    parser.add_argument("--benchmark", default="urfd")
    parser.add_argument("--variants", nargs="*",
                        default=["baseline_stgcn", "ablation_no_preimpact_loss",
                                 "ours_preimpact"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="frontend/lib/curves.json")
    args = parser.parse_args(argv)

    runs_root = root / args.runs if not Path(args.runs).is_absolute() else Path(args.runs)
    index = runs_root / "runs.csv"
    if not index.exists():
        print(f"no runs.csv in {runs_root}")
        return 1

    with index.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curves = {}

    for variant in args.variants:
        stem = variant.replace("ablation_", "")
        matching = [
            Path(r["run_dir"]) for r in runs
            if r.get("config") == stem and int(float(r.get("seed") or 0)) == args.seed
        ]
        if not matching:
            print(f"{variant}: no runs, skipped")
            continue

        print(f"\n{variant}: pooling {len(matching)} folds")
        items, w_pre = pooled_scores(sorted(matching), device)
        if not items or w_pre is None:
            print(f"{variant}: nothing scored")
            continue

        rows = sweep(items, w_pre)
        total_falls = rows[0]["num_falls"] if rows else 0
        negative_minutes = round(
            sum(int((it.labels == 0).sum()) for it in items)
            / float(np.mean([it.clip.fps for it in items])) / 60.0, 2
        )
        curves[stem] = {
            "variant": stem,
            "folds": len(matching),
            "falls": total_falls,
            "negative_minutes": negative_minutes,
            "points": rows,
        }
        print(f"  {len(rows)} operating points over {total_falls} falls "
              f"and {negative_minutes} min of normal activity")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": args.benchmark, "curves": curves}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
