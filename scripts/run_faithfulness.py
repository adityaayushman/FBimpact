"""E1 (PLAN_COMPLETION.txt): re-score saved checkpoints with every explanation method.

    python scripts/run_faithfulness.py --results results/combined_confirm \
        --arms combined combined_phases \
        --methods attention gradient gradient_input occlusion

No retraining. For each run of the chosen arms, the saved `best.pt` is evaluated on its own
test clips with `--explain --set explain.method=<method>` and the result is written to
`<run dir>/faith_<method>/`, so nothing the original run wrote is touched. Finished
(run, method) pairs are skipped, so the script can be restarted freely.

A CUDA error leaves a process unusable for every later call, so on one the script exits with
status 3 and the caller starts a fresh process (the runner does this) rather than letting
every remaining pair fail one after another.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval as eval_module  # noqa: E402


def runs_of(results: Path, arms: list[str]) -> list[Path]:
    path = results / "runs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    found = []
    for row in rows:
        if row["config"] in arms:
            run_dir = Path(row["run_dir"])
            found.append(run_dir if run_dir.is_absolute() else ROOT / run_dir)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results/combined_confirm")
    parser.add_argument("--arms", nargs="+", default=["combined", "combined_phases"])
    parser.add_argument("--methods", nargs="+", default=["attention", "gradient", "gradient_input", "occlusion"])
    args = parser.parse_args(argv)

    runs = runs_of(ROOT / args.results, args.arms)
    todo = [(r, m) for r in runs for m in args.methods
            if not (r / f"faith_{m}" / "faithfulness_test.json").exists()]
    print(f"{len(runs)} runs x {len(args.methods)} methods; {len(todo)} pairs still to do", flush=True)

    failures = 0
    for i, (run, method) in enumerate(todo, 1):
        out = run / f"faith_{method}"
        started = time.time()
        try:
            eval_module.main(["--checkpoint", str(run / "best.pt"), "--split", "test", "--explain",
                              "--set", f"explain.method={method}", "--out", str(out)])
            print(f"[{i}/{len(todo)}] {run.name} {method}: ok in {time.time() - started:.0f}s", flush=True)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAILED: {run.name} {method}: {error}", flush=True)
            traceback.print_exc()
            if "CUDA" in str(error) or "out of memory" in str(error).lower():
                print("CUDA state is unusable in this process; exiting so a fresh one can continue", flush=True)
                return 3
    print(f"done: {len(todo) - failures} ok, {failures} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
