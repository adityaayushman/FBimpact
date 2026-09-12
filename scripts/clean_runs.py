"""Remove run directories that produced no result.

    python scripts/clean_runs.py --dry-run
    python scripts/clean_runs.py

A grid run with `--keep-going` creates a directory for every attempt, including
the ones that crash - and a crashed attempt still leaves its config, logs and
sometimes a partial checkpoint behind. Those directories are never referenced by
`runs.csv`, which only gains a row when a run completes and is evaluated, so the
set difference between "directories present" and "directories referenced" is
exactly the set of failures.

This matters beyond tidiness: `scripts/export_curve.py` walks run directories to
pool per-fold checkpoints, and a half-written directory from a crash is a
plausible-looking input that would either fail late or, worse, contribute a
checkpoint that was never evaluated.

Directories referenced by `runs.csv` are always kept: their checkpoints are what
the operating-point curves are built from.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def referenced_runs(benchmark: Path) -> set[str]:
    """Directory names that `runs.csv` records a completed result for."""
    index = benchmark / "runs.csv"
    if not index.exists():
        return set()
    with index.open(newline="", encoding="utf-8") as handle:
        return {
            Path(row["run_dir"]).name
            for row in csv.DictReader(handle)
            if (row.get("run_dir") or "").strip()
        }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be removed and stop")
    parser.add_argument("--keep-loose", action="store_true",
                        help="keep run directories sitting outside a benchmark folder; "
                             "those are usually smoke tests left behind by hand")
    args = parser.parse_args(argv)

    results = root / args.results
    if not results.exists():
        print(f"no results directory at {results}")
        return 0

    removed = kept = 0
    freed = 0

    for entry in sorted(results.iterdir()):
        if not entry.is_dir():
            continue

        # A benchmark folder holds runs.csv; anything else at this level is a
        # loose run directory, typically a hand-run smoke test.
        if not (entry / "runs.csv").exists():
            if args.keep_loose or (entry / "best.pt").exists() is False:
                # No runs.csv and no checkpoint - an empty or abandoned shell.
                pass
            size = directory_size(entry)
            if args.keep_loose:
                print(f"  keep (loose)  {entry.name}")
                kept += 1
                continue
            print(f"  {'would remove' if args.dry_run else 'removing'} loose "
                  f"{entry.name}  ({size / 1e6:.1f} MB)")
            if not args.dry_run:
                shutil.rmtree(entry, ignore_errors=True)
            removed += 1
            freed += size
            continue

        keep = referenced_runs(entry)
        for run in sorted(d for d in entry.iterdir() if d.is_dir()):
            if run.name in keep:
                kept += 1
                continue
            size = directory_size(run)
            print(f"  {'would remove' if args.dry_run else 'removing'} "
                  f"{entry.name}/{run.name}  ({size / 1e6:.1f} MB)")
            if not args.dry_run:
                shutil.rmtree(run, ignore_errors=True)
            removed += 1
            freed += size

    verb = "would free" if args.dry_run else "freed"
    print(f"\n{removed} unreferenced run directories, {kept} kept, {verb} {freed / 1e6:.1f} MB")
    if args.dry_run and removed:
        print("re-run without --dry-run to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
