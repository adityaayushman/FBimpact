"""Turn completed runs into the JSON the website renders.

    python scripts/export_results.py

Reads the ablation tables written by `scripts/run_ablations.py` and emits
`frontend/lib/results.json`. The website imports that file directly, so every
number on the site traces to a run directory on disk rather than being typed
into a component by hand - and regenerating after a new grid is one command.

If a benchmark has not been run, it is simply absent from the output and the
site says so, rather than showing a plausible placeholder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import load_cache, summarise  # noqa: E402

# Display names and ordering for the ablation table. The key is the config file
# *stem* as written into runs.csv by run_ablations.py - so `no_temporal`, from
# configs/ablations/no_temporal.yaml, not the `ablation_no_temporal` run name.
VARIANTS = [
    ("ours_preimpact", "Ours — pre-impact loss + grounding head", "ours"),
    ("baseline_stgcn", "Baseline — ST-GCN, plain classification loss", "baseline"),
    ("no_preimpact_loss", "− pre-impact loss (λ = 0)", "ablation"),
    ("no_temporal", "− temporal modelling", "ablation"),
    ("no_velocity", "− velocity features", "ablation"),
    ("no_grounding", "− grounding head", "ablation"),
]

NUMERIC = [
    "recall", "mean_lead_time", "median_lead_time", "false_alarms_per_hour",
    "specificity", "frame_auc", "frame_f1",
    "faith_deletion_gap", "faith_insertion_gap", "faith_deletion_auc", "faith_insertion_auc",
]


def read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str):
    """Parse a numeric cell, returning None for blanks and non-finite values."""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(value, 5)


def build_benchmark(
    key: str, name: str, kind: str, caveat: str, results_dir: Path, cache_dir: Path
) -> dict | None:
    table = read_table(results_dir / "ablation_table.csv")
    if not table:
        return None

    by_variant = {r.get("variant"): r for r in table}
    rows = []
    for stem, label, role in VARIANTS:
        raw = by_variant.get(stem)
        if not raw:
            continue
        entry: dict = {"variant": stem, "label": label, "role": role,
                       "runs": int(float(raw.get("n_runs") or 0))}
        for metric in NUMERIC:
            entry[metric] = {"mean": num(raw, f"{metric}_mean"), "std": num(raw, f"{metric}_std")}
        rows.append(entry)

    if not rows:
        return None

    dataset = None
    if cache_dir.exists():
        try:
            stats = summarise(load_cache(cache_dir))
            dataset = {
                "clips": stats["clips"], "falls": stats["falls"], "adls": stats["adls"],
                "subjects": stats["subjects"],
                "minutes": round(stats["total_hours"] * 60, 1),
            }
        except (FileNotFoundError, ValueError):
            dataset = None

    seeds = sorted({int(float(r["seed"])) for r in read_table(results_dir / "runs.csv")
                    if (r.get("seed") or "").strip()})

    return {
        "id": key, "name": name, "kind": kind, "caveat": caveat,
        "dataset": dataset, "seeds": seeds, "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="frontend/lib/results.json")
    args = parser.parse_args(argv)

    benchmarks = []

    synthetic = build_benchmark(
        key="synthetic",
        name="Synthetic benchmark",
        kind="synthetic",
        caveat=(
            "Procedurally generated skeletons, not human falls. These numbers show that the "
            "pipeline and the pre-impact objective behave as designed — they say nothing about "
            "performance on real falls. The normal activities are deliberately hard negatives: "
            "sitting down, bending to pick something up and lying down are all controlled descents."
        ),
        results_dir=root / "results" / "bench",
        cache_dir=root / "data" / "cache" / "synthetic_bench",
    )
    if synthetic:
        synthetic["split"] = "Leave-subjects-out, 5 folds (fold 0 reported)"
        benchmarks.append(synthetic)

    urfd = build_benchmark(
        key="urfd",
        name="UR Fall Detection",
        kind="real",
        caveat=(
            "Real video of acted falls (Kwolek & Kepski). The impact frame comes from the "
            "dataset's own per-frame annotation — the first frame marked 'lying' — never from "
            "the skeleton the model consumes, which would make lead time circular. That label "
            "lands at or after true ground contact, so it shortens measured lead time rather "
            "than inflating it. UR Fall publishes no per-sequence subject identity, so splits "
            "are sequence-independent, a weaker guarantee than subject-independent."
        ),
        results_dir=root / "results" / "urfd",
        cache_dir=root / "data" / "cache" / "urfd",
    )
    if urfd:
        urfd["split"] = "Sequence-independent, 5 folds (fold 0 reported)"
        benchmarks.append(urfd)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmarks": benchmarks,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not benchmarks:
        print(f"no completed benchmarks found — wrote an empty {out}")
        print("run scripts/run_ablations.py first")
        return 0

    print(f"wrote {out}")
    for b in benchmarks:
        d = b.get("dataset")
        print(f"  {b['name']}: {len(b['rows'])} variants, seeds {b['seeds']}"
              + (f", {d['clips']} clips / {d['falls']} falls / {d['subjects']} groups" if d else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
