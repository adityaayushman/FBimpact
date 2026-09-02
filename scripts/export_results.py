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

# Summed rather than averaged. A per-hour rate over a few minutes of negative
# time is quantised so coarsely that it invites over-reading: on UR Fall one
# single false trigger moves the rate by about 52/hour. Publishing the absolute
# counts beside the rate is what stops "100 false alarms per hour" being read as
# a hundred events rather than the twelve it actually is.
TOTALS = ["num_false_alarms", "num_falls", "num_warned", "negative_hours"]


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

    # Per-run rows, so totals can be summed rather than read off an average.
    runs = read_table(results_dir / "runs.csv")
    per_variant: dict[str, list[dict]] = {}
    for run in runs:
        per_variant.setdefault(str(run.get("config")), []).append(run)

    rows = []
    for stem, label, role in VARIANTS:
        raw = by_variant.get(stem)
        if not raw:
            continue
        entry: dict = {"variant": stem, "label": label, "role": role,
                       "runs": int(float(raw.get("n_runs") or 0))}
        for metric in NUMERIC:
            entry[metric] = {"mean": num(raw, f"{metric}_mean"), "std": num(raw, f"{metric}_std")}

        for metric in TOTALS:
            values = [num(r, metric) for r in per_variant.get(stem, [])]
            present = [v for v in values if v is not None]
            entry[metric] = round(sum(present), 4) if present else None
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

    # Real benchmarks first: they are what the claims rest on, and a reader
    # landing on a procedurally generated fixture would reasonably assume the
    # numbers describe human falls.
    DEFINITIONS = [
        dict(
            key="le2i", name="Le2i / ImViA", kind="real",
            results="le2i", cache="le2i",
            split="Clip-independent, 5 folds pooled (every clip tested once)",
            caveat=(
                "Real video of acted falls across two scenes, and the largest real benchmark "
                "here at 99 falls against UR Fall's 30. The impact frame is the annotated fall "
                "END frame — from the dataset's own annotation, never from the skeleton the "
                "model consumes, which would make lead time circular. Three falls have too few "
                "frames before impact for a full window to exist; they are kept as guaranteed "
                "misses rather than dropped, which costs about three points of recall equally "
                "across variants. Le2i publishes no clip-to-actor mapping, so splits are "
                "clip-independent — weaker than subject-independent."
            ),
        ),
        dict(
            key="urfd", name="UR Fall Detection", kind="real",
            results="urfd", cache="urfd",
            split="Sequence-independent, 5 folds pooled (every clip tested once)",
            caveat=(
                "Real video of acted falls (Kwolek & Kepski). The impact frame is the first "
                "frame the dataset's own annotation marks 'lying' — never derived from the "
                "skeleton the model consumes. That label lands at or after true ground contact, "
                "so it shortens measured lead time rather than inflating it. With only 30 falls, "
                "per-fold recall ranges from 0.12 to 1.00 and the fold spread is wider than "
                "every difference between variants. UR Fall publishes no per-sequence subject "
                "identity, so splits are sequence-independent."
            ),
        ),
        dict(
            key="synthetic", name="Synthetic fixture", kind="synthetic",
            results="bench", cache="synthetic_bench",
            split="Leave-subjects-out, 5 folds (fold 0 reported)",
            caveat=(
                "Procedurally generated skeletons, not human falls. These numbers show the "
                "pipeline and objective behave as designed and say nothing about performance on "
                "real falls — near-ceiling recall leaves the variants nothing to separate them. "
                "The normal activities are deliberately hard negatives: sitting, bending and "
                "lying down are all controlled descents."
            ),
        ),
    ]

    benchmarks = []
    for spec in DEFINITIONS:
        built = build_benchmark(
            key=spec["key"], name=spec["name"], kind=spec["kind"], caveat=spec["caveat"],
            results_dir=root / "results" / spec["results"],
            cache_dir=root / "data" / "cache" / spec["cache"],
        )
        if built:
            built["split"] = spec["split"]
            benchmarks.append(built)

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
