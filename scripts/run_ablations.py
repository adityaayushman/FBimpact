"""Section 16 - run the ablation grid and build the results table.

    python scripts/run_ablations.py --seeds 0 1 2
    python scripts/run_ablations.py --configs configs/ours_preimpact.yaml configs/baseline_stgcn.yaml
    python scripts/run_ablations.py --seeds 0 --set train.epochs=3   # smoke test

Every variant is trained from scratch on the same split with the same seeds and
evaluated through the same decision logic, and every cell of the output table is
a mean +/- standard deviation over seeds rather than a single run (Section 14).
A one-run ablation table cannot distinguish a real effect from seed noise, which
is exactly the claim these rows are making.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as eval_module  # noqa: E402
import train as train_module  # noqa: E402
from utils.logging import aggregate_seeds, format_table, save_csv, save_json  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/baseline_stgcn.yaml",
    "configs/ours_preimpact.yaml",
    "configs/ablations/no_temporal.yaml",
    "configs/ablations/no_velocity.yaml",
    "configs/ablations/no_preimpact_loss.yaml",
    "configs/ablations/no_grounding.yaml",
]

REPORT_COLUMNS = [
    "recall",
    "mean_lead_time",
    "median_lead_time",
    "false_alarms_per_hour",
    "specificity",
    "frame_auc",
    "frame_f1",
]

FAITHFULNESS_COLUMNS = ["deletion_auc", "insertion_auc", "deletion_gap", "insertion_gap"]


def load_completed(path: Path) -> list[dict]:
    """Rows from a previous invocation's `runs.csv`, for `--resume`.

    A full grid is 18 trainings and takes hours, so losing it to an interrupted
    shell is expensive and entirely avoidable. Numeric columns are restored to
    floats because the aggregation reads them as numbers.
    """
    if not path.exists():
        return []
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key, value in list(row.items()):
            if key in ("config", "run_dir"):
                continue
            text = (value or "").strip()
            if not text:
                row[key] = None
                continue
            try:
                row[key] = float(text)
            except ValueError:
                pass  # genuinely non-numeric (a nested dict written as JSON)
    return rows


def run_one(config: str, seed: int, overrides: list[str], out_root: str, explain: bool) -> dict:
    """Train and evaluate one (config, seed) pair; returns its flat result row."""
    run_dir = train_module.main(
        ["--config", config, "--seed", str(seed), "--out", out_root, "--set", *overrides]
    )
    argv = ["--checkpoint", str(run_dir / "best.pt"), "--split", "test"]
    if explain:
        argv.append("--explain")
    results = eval_module.main(argv)

    row = {"config": Path(config).stem, "seed": seed, "run_dir": str(run_dir)}
    row.update({k: v for k, v in results["report"].items() if not isinstance(v, (dict, list))})
    if "faithfulness" in results:
        row.update({f"faith_{k}": v for k, v in results["faithfulness"].items()
                    if not isinstance(v, (dict, list))})
    return row


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides applied to every run")
    parser.add_argument("--out", default="results/ablations")
    parser.add_argument("--no-explain", action="store_true",
                        help="skip Stage E/F, which dominates evaluation time")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failed run instead of stopping")
    parser.add_argument("--resume", action="store_true",
                        help="skip (config, seed) pairs already present in runs.csv")
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    if args.resume:
        rows = load_completed(out_root / "runs.csv")
        done = {(str(r["config"]), int(float(r["seed"]))) for r in rows if r.get("seed") is not None}
        if done:
            print(f"resuming: {len(done)} run(s) already complete, skipping those\n")

    failures: list[dict] = []
    total = len(args.configs) * len(args.seeds)
    index = 0

    for config in args.configs:
        for seed in args.seeds:
            index += 1
            if (Path(config).stem, seed) in done:
                print(f"[{index}/{total}] {config} seed={seed} — already done, skipping")
                continue
            print(f"\n{'=' * 78}\n[{index}/{total}] {config} seed={seed}\n{'=' * 78}")
            try:
                rows.append(
                    run_one(config, seed, args.set, str(out_root), not args.no_explain)
                )
            except Exception as exc:
                failures.append({"config": config, "seed": seed, "error": repr(exc)})
                print(f"FAILED: {config} seed={seed}")
                traceback.print_exc(limit=3)
                if not args.keep_going:
                    raise

            # Written after every run, so an interrupted grid still leaves usable
            # partial results rather than nothing.
            save_csv(rows, out_root / "runs.csv")

    # -- aggregate across seeds ------------------------------------------------
    columns = REPORT_COLUMNS + [f"faith_{c}" for c in FAITHFULNESS_COLUMNS]
    summary: list[dict] = []
    for config in args.configs:
        name = Path(config).stem
        matching = [r for r in rows if r["config"] == name]
        if not matching:
            continue
        present = [c for c in columns if any(c in r for r in matching)]
        summary.append({"variant": name, **aggregate_seeds(matching, present)})

    save_csv(summary, out_root / "ablation_table.csv")
    save_json({"runs": rows, "summary": summary, "failures": failures},
              out_root / "ablations.json")

    display = [
        {
            "variant": row["variant"],
            "n": row.get("n_runs", 0),
            "recall": f"{row.get('recall_mean', float('nan')):.3f} +/- {row.get('recall_std', 0):.3f}",
            "lead_time_s": f"{row.get('mean_lead_time_mean', float('nan')):.3f} +/- {row.get('mean_lead_time_std', 0):.3f}",
            "FA/h": f"{row.get('false_alarms_per_hour_mean', float('nan')):.2f} +/- {row.get('false_alarms_per_hour_std', 0):.2f}",
            "frame_auc": f"{row.get('frame_auc_mean', float('nan')):.3f}",
            "del_gap": f"{row.get('faith_deletion_gap_mean', float('nan')):+.3f}",
            "ins_gap": f"{row.get('faith_insertion_gap_mean', float('nan')):+.3f}",
        }
        for row in summary
    ]
    print(f"\n{'=' * 78}\nAblation table (mean +/- std over {len(args.seeds)} seeds)\n")
    print(format_table(display))
    if failures:
        print(f"\n{len(failures)} run(s) failed; see ablations.json")
    print(f"\nwritten to {out_root / 'ablation_table.csv'}")
    return out_root


if __name__ == "__main__":
    main()
