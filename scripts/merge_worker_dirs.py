"""Merge the per-worker result folders of a Colab stage into one ordinary results folder.

    python scripts/merge_worker_dirs.py --workers results/combined_rq3_workers --out results/combined_rq3

On Colab each parallel worker writes to its own folder (two processes appending to one
runs.csv would overwrite each other). Afterwards this moves every finished run into `--out` and
writes one `runs.csv` there whose `run_dir` is a repo-relative, forward-slash path
(`results/<out name>/<run>`), which every analysis script and the Windows import understand.

Safe to run repeatedly: runs already moved are skipped, and a (config, seed, fold) that appears
twice keeps the newest. Only rows whose run folder exists are kept. No torch.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def leaf(run_dir: str) -> str:
    return Path(run_dir.replace("\\", "/")).name


def merge(workers: Path, out: Path, results_name: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    name = results_name or out.name
    merged: dict[tuple, dict] = {}
    fields: list[str] = []

    existing = out / "runs.csv"
    if existing.exists():
        with existing.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            for r in reader:
                merged[(r["config"], int(float(r["seed"])), int(float(r.get("fold") or 0)))] = r

    moved = skipped = 0
    for w in sorted(p for p in workers.iterdir() if p.is_dir()) if workers.exists() else []:
        csv_path = w / "runs.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:
                if col not in fields:
                    fields.append(col)
            for r in reader:
                run = leaf(r["run_dir"])
                src, dst = w / run, out / run
                if not src.exists() and not dst.exists():
                    continue                                   # row without a folder: not a usable run
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
                else:
                    skipped += 1
                r["run_dir"] = f"results/{name}/{run}"
                merged[(r["config"], int(float(r["seed"])), int(float(r.get("fold") or 0)))] = r

    rows = [merged[k] for k in sorted(merged)]
    with existing.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return {"runs": len(rows), "moved": moved, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    r = merge(Path(args.workers), Path(args.out))
    print(f"merged into {args.out}: {r['runs']} runs ({r['moved']} moved, {r['skipped']} already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
