"""One-screen progress report for the long-running experiment jobs.

    python scripts/progress.py            # print once
    python scripts/progress.py --watch 60 # refresh every 60 s until Ctrl+C

Reads only `runs.csv` and the run directories' timestamps, so it is safe to run
while training (no torch, a few MB). The ETA assumes the job keeps running without
a gap; it uses the median duration of recent runs and says so when it has too few.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOBS = [
    ("Confirmation grid (PLAN.txt)", ROOT / "results/combined_confirm", 85,
     "4 arms x seeds 1-4 x 5 folds, plus arm D seed 0"),
    ("Held-out-domain runs (PLAN_XDOMAIN.txt)", ROOT / "results/xdomain", 24,
     "4 protocols x 2 arms x seeds 1-3"),
]
MAX_PLAUSIBLE_RUN_SECONDS = 3 * 3600  # longer than this means the job was down, not training


def read_rows(directory: Path) -> list[dict]:
    path = directory / "runs.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_seconds(row: dict) -> float | None:
    """Wall time of one run: from the start stamp in its directory name to its last write."""
    run_dir = Path(row.get("run_dir", ""))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    try:
        # The stamp in a run's directory name is UTC; file times are local.
        started = datetime.strptime(run_dir.name.rsplit("_", 1)[-1], "%Y%m%d-%H%M%S").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    if not run_dir.exists():
        return None
    seconds = run_dir.stat().st_mtime - started
    return seconds if 0 < seconds < MAX_PLAUSIBLE_RUN_SECONDS else None


def training_processes() -> list[str]:
    """Command lines of running experiment jobs (empty list if none or unknown)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
             "'run_ablations|run_xdomain|run_confirm' } | ForEach-Object { $_.Name + '|' + $_.CommandLine }"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    # The query's own PowerShell process contains the search words in its command
    # line, so it must be excluded or it reports itself as a running job.
    return [line for line in out.splitlines() if line.strip() and "Get-CimInstance" not in line]


def fmt_duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m" if hours else f"{rest // 60}m"


def report() -> str:
    lines = [f"Progress at {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    for title, directory, expected, detail in JOBS:
        rows = read_rows(directory)
        done = len(rows)
        bar = "#" * round(30 * done / expected) + "-" * (30 - round(30 * done / expected))
        lines.append(f"{title}")
        lines.append(f"  [{bar}] {done}/{expected}  ({100 * done / expected:.0f}%)   {detail}")
        if rows:
            by = Counter((int(float(r["seed"])), r["config"]) for r in rows)
            lines.append("  finished: " + "; ".join(
                f"seed {s} {c.replace('combined_', '').replace('xd_', '')}: {n}"
                for (s, c), n in sorted(by.items())))
            durations = [d for d in (run_seconds(r) for r in rows[-12:]) if d]
            if len(durations) >= 3 and done < expected:
                typical = statistics.median(durations)
                lines.append(f"  typical run {fmt_duration(typical)}  ->  about "
                             f"{fmt_duration(typical * (expected - done))} left if it never pauses")
            elif done < expected:
                lines.append("  too few timed runs for an ETA")
            newest = max((directory / "runs.csv").stat().st_mtime, 0)
            lines.append(f"  last result written {fmt_duration(time.time() - newest)} ago")
        else:
            lines.append("  no results yet")
        lines.append("")

    procs = training_processes()
    names = sorted({p.split("|")[0] for p in procs})
    working = any("run_ablations" in p and "python" in p.lower() for p in procs)
    lines.append("Job processes: " + ("TRAINING (python run_ablations is alive)" if working
                 else "NOT RUNNING - nothing is training"))
    if procs and not working:
        lines.append("  (only waiting scripts are alive: " + ", ".join(names) + ")")
    if not working:
        lines.append("  restart the grid with:  powershell -File d:\\tmp\\run_all.ps1")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                        help="refresh every N seconds until interrupted")
    args = parser.parse_args()
    if not args.watch:
        print(report())
        return 0
    try:
        while True:
            print("\033[2J\033[H" + report(), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
