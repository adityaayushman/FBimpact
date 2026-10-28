"""Prove that the published results are real: recompute them from the run folders on disk.

    python scripts/verify_results.py                 # prints the ledger, exits 1 if any check fails
    python scripts/verify_results.py --out results/analysis/VERIFICATION.txt

Every number the site and the progress report state is read here from a different place than the
one that produced it. The analysis scripts work from each run's per-clip `outcomes_test.csv`; this
script works from the per-run totals in `runs.csv` and from the saved run folders, and compares.

  1. UR Fall      pooled recall per variant against the figures published in the README.
  2. Confirmation the 80 pre-registered runs are all present, and arm means recomputed from
                  runs.csv equal results/analysis/arms_summary.csv.
  3. Held-out     all 24 runs exist, each tested only on its held-out domain, per-run recall and
                  false alarms equal xdomain_by_run.csv, in-domain recall equals the site's figure.
  4. Explanations the pooled real-data counts and means equal frontend/lib/faithfulness.json.
  5. Run folders  every run has its saved config, environment record, checkpoint and outcomes.
  6. Pre-registration PLAN.txt and PLAN_XDOMAIN.txt were committed before the first run they
                  govern, and have only been extended (no line removed) since.

No torch. Nothing is written except the optional --out file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_faithfulness as af  # noqa: E402
from verify_xdomain import DOMAINS, protocol_of  # noqa: E402

RESULTS = ROOT / "results"
README_UR_FALL = {"no_preimpact_loss": (0.767, 15), "no_grounding": (0.767, 5), "baseline_stgcn": (0.711, 15),
                  "ours_preimpact": (0.633, 15), "no_velocity": (0.533, 5), "no_temporal": (0.333, 5)}
CONFIRM_ARMS = ["combined_xyv", "combined", "combined_phases", "combined_nocons"]
RUN_FILES = ("config.yaml", "environment.json", "best.pt", "outcomes_test.csv", "split.json")
STAMP = re.compile(r"(\d{8}-\d{6})$")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> float:
    return float(row[key])


def key(row: dict) -> tuple[str, int, int]:
    return row["config"], int(float(row["seed"])), int(float(row.get("fold") or 0))


def pooled_recall(rows: list[dict]) -> float:
    """Warned divided by tested, summed over runs: the same pooling the README states."""
    return sum(num(r, "num_warned") for r in rows) / sum(num(r, "num_falls") for r in rows)


def run_dir_of(results_dir: Path, row: dict) -> Path:
    return results_dir / Path(row["run_dir"].replace("\\", "/")).name


def started_utc(run_dir: Path) -> datetime:
    return datetime.strptime(STAMP.search(run_dir.name).group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


def removed_lines(diff_text: str) -> list[str]:
    """Lines a diff deletes (a plan may be extended with addenda, never rewritten)."""
    return [ln for ln in diff_text.splitlines() if ln.startswith("-") and not ln.startswith("---")]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


class Ledger:
    def __init__(self) -> None:
        self.items: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        self.items.append((bool(ok), what, detail))

    @property
    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.items if not ok)


def check_ur_fall(L: Ledger) -> None:
    rows = read_rows(RESULTS / "urfd" / "runs.csv")
    for variant, (claimed, runs) in README_UR_FALL.items():
        mine = [r for r in rows if r["config"] == variant]
        got = pooled_recall(mine)
        L.check(len(mine) == runs and abs(got - claimed) <= 0.0006,
                f"UR Fall {variant}: pooled recall {claimed:.3f} over {runs} runs",
                f"recomputed {got:.4f} over {len(mine)} runs")


def check_confirmation(L: Ledger) -> None:
    confirm = RESULTS / "combined_confirm"
    rows = [r for r in read_rows(confirm / "runs.csv") if int(float(r["seed"])) in (1, 2, 3, 4)]
    keys = {key(r) for r in rows}
    planned = {(a, s, f) for a in CONFIRM_ARMS for s in (1, 2, 3, 4) for f in range(5)}
    L.check(keys == planned and len(rows) == 80, "Confirmation grid: all 80 pre-registered runs present, none extra",
            f"{len(keys & planned)} of {len(planned)} planned, {len(rows)} rows")
    summary = {r["arm"]: r for r in read_rows(RESULTS / "analysis" / "arms_summary.csv")}
    for arm in CONFIRM_ARMS:
        recalls, fas = [], []
        for seed in (1, 2, 3, 4):
            cell = [r for r in rows if r["config"] == arm and int(float(r["seed"])) == seed]
            recalls.append(pooled_recall(cell))
            fas.append(sum(num(r, "num_false_alarms") for r in cell))
        mean_recall, mean_fa = sum(recalls) / 4, sum(fas) / 4
        pub = summary[arm]
        L.check(abs(mean_recall - float(pub["recall_mean"])) <= 6e-5 and abs(mean_fa - float(pub["fa_mean"])) <= 0.01,
                f"Confirmation {arm}: recall {float(pub['recall_mean']):.4f}, false alarms {float(pub['fa_mean']):.2f}",
                f"recomputed from runs.csv: recall {mean_recall:.4f}, false alarms {mean_fa:.2f}")


def check_heldout(L: Ledger) -> None:
    xd = RESULTS / "xdomain"
    rows = read_rows(xd / "runs.csv")
    L.check(len(rows) == 24 and len({key(r) for r in rows}) == 24, "Held-out domains: 24 runs (4 protocols x 2 arms x 3 seeds)",
            f"{len(rows)} rows")
    wrong = []
    for r in rows:
        run = run_dir_of(xd, r)
        mode = __import__("yaml").safe_load((run / "config.yaml").read_text(encoding="utf-8"))["data"]["split"].get("mode")
        ids = [c.get("clip_id") or c.get("clip") for c in read_rows(run / "outcomes_test.csv")]
        inside = DOMAINS[protocol_of(r["config"])]
        if mode != "holdout" or not ids or any(i and not inside(i) for i in ids):
            wrong.append(run.name)
    L.check(not wrong, "Held-out domains: every run tested only on its held-out domain", f"{len(rows) - len(wrong)} of {len(rows)} verified" + (f"; wrong: {wrong[:2]}" if wrong else ""))
    published = {(r["protocol"], r["arm"], int(float(r["seed"]))): r for r in read_rows(RESULTS / "analysis" / "xdomain_by_run.csv")}
    mismatches = 0
    for r in rows:
        protocol = protocol_of(r["config"])
        arm = r["config"].rsplit("_", 1)[1]
        pub = published.get((protocol, arm, int(float(r["seed"]))))
        if pub is None or abs(num(r, "num_warned") / num(r, "num_falls") - float(pub["recall"])) > 1e-3 \
                or abs(num(r, "num_false_alarms") - float(pub["false_alarms"])) > 1e-9:
            mismatches += 1
    L.check(mismatches == 0, "Held-out domains: per-run recall and false alarms equal xdomain_by_run.csv", f"{24 - mismatches} of 24 match")
    gen = json.loads((ROOT / "frontend" / "lib" / "generalisation.json").read_text(encoding="utf-8"))
    claimed = {a: v["in_domain"]["recall"] for p in gen["protocols"][:1] for a, v in p["arms"].items()}
    confirm = read_rows(RESULTS / "combined_confirm" / "runs.csv")
    for arm, config in (("ctrl", "combined_xyv"), ("phases", "combined_phases")):
        cell = [r for r in confirm if r["config"] == config and int(float(r["seed"])) in (1, 2, 3)]
        got = pooled_recall(cell)
        L.check(len(cell) == 15 and abs(got - claimed[arm]) <= 6e-5, f"In-domain recall {claimed[arm]:.4f} for the {arm} arm (seeds 1-3, 15 runs)",
                f"recomputed {got:.4f} over {len(cell)} runs")


def check_faithfulness(L: Ledger) -> None:
    real = af.real_data(af.collect(RESULTS))
    weights = [r["n"] for r in real]
    published = json.loads((ROOT / "frontend" / "lib" / "faithfulness.json").read_text(encoding="utf-8"))["overall"]
    ok = len(real) == published["runs"] and sum(weights) == published["explanations"]
    d = af.weighted_mean(__import__("numpy").array([r["deletion_gap"] for r in real]), __import__("numpy").array(weights, float))
    i = af.weighted_mean(__import__("numpy").array([r["insertion_gap"] for r in real]), __import__("numpy").array(weights, float))
    ok = ok and abs(d - published["deletion"]["mean"]) < 1e-9 and abs(i - published["insertion"]["mean"]) < 1e-9
    L.check(ok, f"Explanations: {published['runs']} real-data runs, {published['explanations']} explanations, gaps {published['deletion']['mean']:+.3f} / {published['insertion']['mean']:+.3f}",
            f"recomputed {len(real)} runs, {sum(weights)} explanations, gaps {d:+.3f} / {i:+.3f}")


def check_run_folders(L: Ledger) -> None:
    for name in ("urfd", "le2i", "combined", "combined_confirm", "xdomain", "bench"):
        folder = RESULTS / name
        rows = read_rows(folder / "runs.csv")
        missing = [(run_dir_of(folder, r).name, f) for r in rows for f in RUN_FILES if not (run_dir_of(folder, r) / f).exists()]
        L.check(not missing, f"Run folders results/{name}: {len(rows)} runs each have config, environment record, checkpoint, per-clip outcomes",
                f"{len(missing)} missing files" + (f", e.g. {missing[0]}" if missing else ""))


def check_preregistration(L: Ledger) -> None:
    plans = {
        "PLAN.txt": [RESULTS / "combined_confirm"],
        "PLAN_XDOMAIN.txt": [RESULTS / "xdomain", RESULTS / "xdomain_invalid_fold0"],
    }
    for plan, folders in plans.items():
        rel = f"results/analysis/{plan}"
        first = git("log", "--diff-filter=A", "--format=%H", "--", rel).splitlines()[-1]
        committed = datetime.fromisoformat(git("show", "-s", "--format=%cI", first)).astimezone(timezone.utc)
        starts = []
        for folder in folders:
            for r in read_rows(folder / "runs.csv") if (folder / "runs.csv").exists() else []:
                if plan == "PLAN.txt" and int(float(r["seed"])) == 0:
                    continue                                   # seed 0 generated the hypothesis, it is descriptive only
                starts.append(started_utc(run_dir_of(folder, r)))
            if plan == "PLAN_XDOMAIN.txt":
                starts += [started_utc(d) for d in folder.glob("xd_*_seed*") if STAMP.search(d.name)]
        first_run = min(starts)
        margin = (first_run - committed).total_seconds()
        L.check(margin > 0, f"{plan} was committed before the first run it governs",
                f"committed {committed:%Y-%m-%d %H:%M:%S} UTC, first run started {first_run:%Y-%m-%d %H:%M:%S} UTC ({margin:,.0f} s later)")
        deleted = removed_lines(git("diff", first, "HEAD", "--", rel))
        L.check(not deleted, f"{plan} has only been extended since it was committed", f"{len(deleted)} lines removed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None, help="also write the ledger to this file")
    args = parser.parse_args(argv)

    L = Ledger()
    for check in (check_ur_fall, check_confirmation, check_heldout, check_faithfulness, check_run_folders, check_preregistration):
        check(L)

    lines = [f"Results verification, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, code commit {git('rev-parse', '--short', 'HEAD')}", ""]
    for ok, what, detail in L.items:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {what}")
        if detail:
            lines.append(f"       {detail}")
    lines += ["", f"{len(L.items) - L.failed} of {len(L.items)} checks passed"]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 1 if L.failed else 0


if __name__ == "__main__":
    sys.exit(main())
