"""Generate colab/fbimpact_research.ipynb, pinned to one git commit (PLAN_COMPLETION.txt).

    python scripts/build_colab_notebook.py --commit <40-char sha>

The notebook runs E1 (160 explanation re-scores), E3 (40 trainings) and E2 (180 trainings) on a
Colab GPU from the pinned commit, on skeleton data it has checked against a sha256 manifest, and
returns one `results_for_import.zip`, which `scripts/import_colab_results.py` then gates.

It is resumable: progress is copied to Google Drive every two minutes and restored on the next
run, and every stage skips what is already finished. After a disconnect, run all cells again.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "fbimpact_research.ipynb"

INTRO = """# Pre-impact fall anticipation: the three pre-registered experiments

Runs **E1** (explanation methods), **E3** (pre-impact loss on the combined benchmark) and **E2**
(ablation grids) from `results/analysis/PLAN_COMPLETION.txt`, on this machine, from one pinned code commit.

**Before you start**
1. Put `fbimpact_data_bundle.zip` and `fbimpact_checkpoints.zip` in the Drive folder `My Drive/fbimpact/`.
2. *Runtime → Change runtime type → a GPU* (T4 is fine).
3. *Runtime → Run all*. Allow the Drive pop-up.

**If it disconnects** (free Colab does that), open the notebook again and *Run all*. It picks up where it stopped.
**When the last cell says DONE**, download `My Drive/fbimpact/results_for_import.zip` and give it to Claude.

Nothing here changes the questions, the decision rules or the configs; it only runs them.
"""

SETTINGS = '''COMMIT = "__COMMIT__"                       # the code every run uses; checked below
REPO_URL = "https://github.com/adityaayushman/FBimpact.git"
DRIVE = "/content/drive/MyDrive/fbimpact"    # where the two zips are, and where progress is kept
REPO = "/content/fbimpact"
N_WORKERS = 3                                # trainings running at once on the one GPU
'''

SETUP = '''import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

def sh(cmd):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\\n{r.stdout}\\n{r.stderr}")
    return r.stdout.strip()

for z in ("fbimpact_data_bundle.zip", "fbimpact_checkpoints.zip"):
    assert Path(DRIVE, z).exists(), f"Upload {z} to {DRIVE} first."

# -- the code: exactly the pinned commit ----------------------------------------------------------
if not Path(REPO, ".git").exists():
    sh(["git", "clone", "--quiet", REPO_URL, REPO])
sh(["git", "-C", REPO, "fetch", "--quiet", "--all"])
sh(["git", "-C", REPO, "checkout", "--quiet", COMMIT])
head = sh(["git", "-C", REPO, "rev-parse", "HEAD"])
assert head == COMMIT, f"checked out {head}, expected {COMMIT}"
print("code commit:", head)

# -- the data: unpack, then prove it is the data the plan names ------------------------------------
with zipfile.ZipFile(Path(DRIVE, "fbimpact_data_bundle.zip")) as z:
    manifest = json.loads(z.read("MANIFEST.json"))
    for name in z.namelist():
        if name != "MANIFEST.json":
            z.extract(name, REPO)
bad = []
for rel, meta in manifest["files"].items():
    h = hashlib.sha256(Path(REPO, rel).read_bytes()).hexdigest()
    if h != meta["sha256"]:
        bad.append(rel)
extra = [p.as_posix() for d in ("data/cache/urfd", "data/cache/le2i") for p in Path(REPO, d).glob("*.npz")
         if p.relative_to(REPO).as_posix() not in manifest["files"]]
assert not bad and not extra, f"data does not match the manifest: changed={bad[:3]} extra={extra[:3]}"
print(f"data: {len(manifest['files'])} skeleton files match the manifest (built {manifest['created_utc']})")

with zipfile.ZipFile(Path(DRIVE, "fbimpact_checkpoints.zip")) as z:
    z.extractall(REPO)

# -- the machine -----------------------------------------------------------------------------------
import torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > pick a GPU, then run all again."
GPU = torch.cuda.get_device_name(0)
print("GPU:", GPU, "| torch", torch.__version__, "| python", sys.version.split()[0])
'''

PERSIST = '''if not shutil.which("rsync"):
    subprocess.run(["apt-get", "install", "-y", "-qq", "rsync"], capture_output=True)

RES = Path(REPO, "results")
DRIVE_RES = Path(DRIVE, "results")
DRIVE_RES.mkdir(parents=True, exist_ok=True)
RSYNC = ["rsync", "-rl", "--no-perms", "--no-owner", "--no-group", "--omit-dir-times"]

# Restore earlier progress. --ignore-existing: never overwrite something this session already has.
subprocess.run(RSYNC + ["--ignore-existing", f"{DRIVE_RES}/", f"{RES}/"], capture_output=True)

def save_now():
    subprocess.run(RSYNC + ["--size-only", f"{RES}/", f"{DRIVE_RES}/"], capture_output=True)

_stop = threading.Event()
def _keep_saving():
    while not _stop.wait(120):
        save_now()
threading.Thread(target=_keep_saving, daemon=True).start()

LOGS = RES / "_logs"
LOGS.mkdir(parents=True, exist_ok=True)

def run_job(name, argv, tries=2):
    """One python process in the repo. Retried in a fresh process: a CUDA error poisons the old one."""
    for attempt in range(1, tries + 1):
        with open(LOGS / f"{name}.log", "a") as log:
            p = subprocess.run([sys.executable, *argv], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if p.returncode == 0:
            return name, 0
    return name, p.returncode

def run_stage(title, jobs, tries=2):
    start, failed = time.time(), []
    print(f"{title}: {len(jobs)} jobs, {N_WORKERS} at a time")
    with ThreadPoolExecutor(N_WORKERS) as pool:
        futures = [pool.submit(run_job, name, argv, tries) for name, argv in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            name, rc = f.result()
            spent = time.time() - start
            print(f"  [{i}/{len(jobs)}] {name}: {'ok' if rc == 0 else f'FAILED (exit {rc}, see results/_logs/{name}.log)'}"
                  f"   {spent / 60:.0f} min elapsed, about {spent / i * (len(jobs) - i) / 60:.0f} min left")
            if rc:
                failed.append(name)
    save_now()
    print(f"{title}: {'all jobs finished' if not failed else f'{len(failed)} job(s) failed: re-run this cell to retry them'}")
    return failed

def merge(workers, out):
    r = subprocess.run([sys.executable, "scripts/merge_worker_dirs.py", "--workers", workers, "--out", out],
                       cwd=REPO, text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr.strip())
    save_now()

def count_runs(workers_dir):
    n = 0
    for csv_path in Path(REPO, workers_dir).glob("*/runs.csv"):
        n += max(0, len(csv_path.read_text().splitlines()) - 1)
    return n

FOLDS = ["0", "1", "2", "3", "4"]
VARIANTS = ["ours_preimpact", "baseline_stgcn", "no_preimpact_loss", "no_temporal", "no_velocity", "no_grounding"]
METHODS = ["attention", "gradient", "gradient_input", "occlusion"]
print("ready; progress is saved to Drive every 2 minutes")
'''

E1 = '''# E1: re-score the 40 saved confirmation checkpoints with every explanation method (160 pairs, no training).
jobs = [(f"e1_{arm}_{m}", ["scripts/run_faithfulness.py", "--results", "results/combined_confirm", "--arms", arm, "--methods", m])
        for arm in ("combined", "combined_phases") for m in METHODS]
run_stage("E1 explanation methods", jobs, tries=4)
done = len(list(RES.glob("combined_confirm/*/faith_*/faithfulness_test.json")))
print(f"E1 re-scores finished: {done} of 160")
'''

E3 = '''# E3: arm B (pre-impact loss off) and arm G (on), seeds 1-4 x 5 folds, on this machine.
OVER = ["train.epochs=30", "train.patience=6", "train.eval_every=2", "train.batch_size=24"]
jobs = [(f"e3_{arm}_s{seed}",
         ["scripts/run_ablations.py", "--configs", f"configs/{arm}.yaml", "--seeds", str(seed), "--folds", *FOLDS,
          "--out", f"results/combined_rq3_workers/{arm}_s{seed}", "--keep-going", "--resume", "--no-explain",
          "--set", *OVER])
        for arm in ("combined", "combined_preimpact") for seed in (1, 2, 3, 4)]
run_stage("E3 pre-impact loss", jobs)
merge("results/combined_rq3_workers", "results/combined_rq3")
if count_runs("results/combined_rq3_workers") >= 40:
    r = subprocess.run([sys.executable, "scripts/matched_recall.py", "--results", "results/combined_rq3", "--arms", "combined",
                        "combined_preimpact", "--seeds", "1", "2", "3", "4", "--out", "results/analysis_rq3/matched_recall.csv"],
                       cwd=REPO, text=True, capture_output=True)
    print(r.stdout[-1500:] or r.stderr[-1500:])
save_now()
print(f"E3 runs finished: {count_runs('results/combined_rq3_workers')} of 40")
'''

E2_URFD = '''# E2: the complete ablation grid on UR Fall, from the frozen configs (6 variants x seeds 0-2 x 5 folds = 90 runs).
def e2_jobs(bench):
    return [(f"e2_{bench}_{v}_s{seed}",
             ["scripts/run_ablations.py", "--configs", f"configs/frozen/{bench}/{v}.yaml", "--seeds", str(seed), "--folds", *FOLDS,
              "--out", f"results/{bench}_colab_workers/{v}_s{seed}", "--keep-going", "--resume"])
            for v in VARIANTS for seed in (0, 1, 2)]

run_stage("E2 UR Fall", e2_jobs("urfd"))
merge("results/urfd_colab_workers", "results/urfd_colab")
print(f"UR Fall runs finished: {count_runs('results/urfd_colab_workers')} of 90")
'''

E2_LE2I = '''# E2 on Le2i (the same 90 runs from its frozen configs).
run_stage("E2 Le2i", e2_jobs("le2i"))
merge("results/le2i_colab_workers", "results/le2i_colab")
print(f"Le2i runs finished: {count_runs('results/le2i_colab_workers')} of 90")
'''

PACKAGE = '''# Check, package and download. best.pt files stay on Drive (results/); the zip carries everything the analysis needs.
for bench in ("urfd", "le2i"):
    merge(f"results/{bench}_colab_workers", f"results/{bench}_colab")
merge("results/combined_rq3_workers", "results/combined_rq3")

ZIP = Path(DRIVE, "results_for_import.zip")
tmp = Path("/content/results_for_import.zip")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("provenance.json", json.dumps({
        "pinned_commit": COMMIT, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu_at_packaging": GPU, "torch": torch.__version__, "python": sys.version.split()[0], "n_workers": N_WORKERS}, indent=1))
    for stage in ("combined_rq3", "urfd_colab", "le2i_colab"):
        for p in sorted(Path(RES, stage).rglob("*")):
            if p.is_file() and p.name != "best.pt":
                z.write(p, p.relative_to(REPO).as_posix())
    for p in sorted(RES.glob("combined_confirm/*/faith_*/*")):
        if p.is_file():
            z.write(p, p.relative_to(REPO).as_posix())
    mr = Path(RES, "analysis_rq3", "matched_recall.csv")
    if mr.exists():
        z.write(mr, mr.relative_to(REPO).as_posix())
    for p in sorted(LOGS.glob("*.log")):
        z.write(p, p.relative_to(REPO).as_posix())
shutil.copy2(tmp, ZIP)
print(f"wrote {ZIP} ({ZIP.stat().st_size / 1e6:.0f} MB)\\n")

# The same gate the analysis machine applies. If it refuses here, nothing is lost: it says what to re-run.
gate = subprocess.run([sys.executable, "scripts/import_colab_results.py", "--zip", str(tmp), "--dry-run", "--tmp", "/content/_gate"],
                      cwd=REPO, text=True, capture_output=True)
print(gate.stdout[-4000:] or gate.stderr[-2000:])
print("DONE: download results_for_import.zip from the fbimpact folder in Google Drive.")
'''


def cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def build(commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit must be a full 40-character git sha")
    cells = [
        cell("markdown", INTRO),
        cell("code", SETTINGS.replace("__COMMIT__", commit)),
        cell("code", SETUP),
        cell("code", PERSIST),
        cell("markdown", "## E1: explanation methods (re-scoring, no training)"),
        cell("code", E1),
        cell("markdown", "## E3: pre-impact loss on the combined benchmark"),
        cell("code", E3),
        cell("markdown", "## E2: ablation grids (UR Fall, then Le2i)"),
        cell("code", E2_URFD),
        cell("code", E2_LE2I),
        cell("markdown", "## Check, package, download"),
        cell("code", PACKAGE),
    ]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"name": "fbimpact_research.ipynb", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--commit", required=True)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    nb = build(args.commit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} pinned to {args.commit[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
