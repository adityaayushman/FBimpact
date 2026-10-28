"""Bring the Colab results home, and refuse anything that is not what the plan said was run.

    python scripts/import_colab_results.py --zip colab_results/results_for_import.zip
    python scripts/import_colab_results.py --zip ... --dry-run        # validate only

PLAN_COMPLETION.txt (addendum 2026-10-28) makes this the gate: no Colab result is analysed until
every run in it passes the checks below.

  1. Config     each run's saved config.yaml agrees with the expected config for its arm (the frozen
                configs, or configs/combined*.yaml plus the confirmation-grid overrides) on every
                setting that affects training, so a stray flag cannot slip through.
  2. Provenance each run's environment.json shows a CUDA device and the pinned git commit.
  3. Coverage   the (config, seed, fold) set is reported against the planned one; missing runs are
                reported, not hidden (a partial import is allowed so progress can be tracked).
  4. E1         every re-score has a faithfulness_test.json whose method matches its folder.

Nothing is copied unless every present run passes. No torch.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["ours_preimpact", "baseline_stgcn", "no_preimpact_loss", "no_temporal", "no_velocity", "no_grounding"]
METHODS = ["attention", "gradient", "gradient_input", "occlusion"]
CONFIRM_OVERRIDES = {"train": {"epochs": 30, "patience": 6, "eval_every": 2, "batch_size": 24}}

STAGES = {
    "combined_rq3": {"seeds": (1, 2, 3, 4), "overrides": CONFIRM_OVERRIDES,
                     "configs": {"combined": "configs/combined.yaml", "combined_preimpact": "configs/combined_preimpact.yaml"}},
    "urfd_colab": {"seeds": (0, 1, 2), "overrides": {}, "configs": {v: f"configs/frozen/urfd/{v}.yaml" for v in VARIANTS}},
    "le2i_colab": {"seeds": (0, 1, 2), "overrides": {}, "configs": {v: f"configs/frozen/le2i/{v}.yaml" for v in VARIANTS}},
}
# Sections that decide what is trained and how it is scored. `run` (seed, paths, name) and
# `data.source` (relabelled at run time) legitimately differ.
CHECKED = ("model", "loss", "features", "labels", "augment", "train", "decision", "explain")


def load_expected(repo: Path, rel: str, overrides: dict) -> dict:
    path = repo / rel
    if "frozen" in rel:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        spec = importlib.util.spec_from_file_location("cfgmod", repo / "utils" / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.load_config(str(path))
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def diff(expected, actual, path: str = "") -> list[str]:
    """Settings present in `expected` that `actual` does not match (extra keys in actual are fine)."""
    if isinstance(expected, dict):
        out = []
        for k, v in expected.items():
            if not isinstance(actual, dict) or k not in actual:
                out.append(f"{path}{k}: expected {v!r}, missing")
            else:
                out += diff(v, actual[k], f"{path}{k}.")
        return out
    return [] if expected == actual else [f"{path[:-1]}: expected {expected!r}, got {actual!r}"]


def relevant(cfg: dict) -> dict:
    """The settings that decide what is trained. `data.split.fold` varies per run and is checked
    against the run's row instead (see `validate_stage`)."""
    out = {k: cfg[k] for k in CHECKED if k in cfg}
    data = cfg.get("data", {})
    split = {k: v for k, v in data.get("split", {}).items() if k != "fold"}
    out["data"] = {"cache_dir": data.get("cache_dir"), "split": split}
    out["run"] = {k: cfg.get("run", {}).get(k) for k in ("device", "deterministic")}
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_stage(repo: Path, root: Path, stage: str, spec: dict, pinned: str) -> dict:
    rows = read_rows(root / "results" / stage / "runs.csv")
    expected_cfgs = {name: relevant(load_expected(repo, rel, spec["overrides"])) for name, rel in spec["configs"].items()}
    planned = {(c, s, f) for c in spec["configs"] for s in spec["seeds"] for f in range(5)}
    seen, problems, gpus, torches = set(), [], set(), set()
    for r in rows:
        key = (r["config"], int(float(r["seed"])), int(float(r.get("fold") or 0)))
        run = Path(r["run_dir"].replace("\\", "/")).name
        run_dir = root / "results" / stage / run
        tag = f"{stage}/{run}"
        if key not in planned:
            problems.append(f"{tag}: {key} is not part of the plan")
            continue
        if not (run_dir / "config.yaml").exists():
            problems.append(f"{tag}: no config.yaml")
            continue
        saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        problems += [f"{tag}: config {d}" for d in diff(expected_cfgs[key[0]], relevant(saved))]
        saved_fold = saved.get("data", {}).get("split", {}).get("fold")
        if saved_fold != key[2]:
            problems.append(f"{tag}: trained on fold {saved_fold}, recorded as fold {key[2]}")
        if saved.get("run", {}).get("seed") != key[1]:
            problems.append(f"{tag}: trained with seed {saved.get('run', {}).get('seed')}, recorded as seed {key[1]}")
        env_path = run_dir / "environment.json"
        if not env_path.exists():
            problems.append(f"{tag}: no environment.json")
        else:
            env = json.loads(env_path.read_text(encoding="utf-8"))
            if not env.get("cuda_available"):
                problems.append(f"{tag}: ran without CUDA")
            if pinned and env.get("git_commit") != pinned:
                problems.append(f"{tag}: ran on commit {str(env.get('git_commit'))[:10]}, pinned {pinned[:10]}")
            gpus.add(env.get("cuda_device")), torches.add(env.get("torch"))
        if not (run_dir / "outcomes_test.csv").exists():
            problems.append(f"{tag}: no outcomes_test.csv")
        seen.add(key)
    return {"stage": stage, "present": len(seen), "planned": len(planned), "missing": sorted(planned - seen),
            "problems": problems, "gpus": sorted(g for g in gpus if g), "torch": sorted(t for t in torches if t)}


def validate_e1(root: Path, local_results: Path) -> dict:
    runs = [r for r in read_rows(local_results / "combined_confirm" / "runs.csv") if r["config"] in ("combined", "combined_phases")]
    problems, present = [], 0
    for r in runs:
        run = Path(r["run_dir"].replace("\\", "/")).name
        for m in METHODS:
            f = root / "results" / "combined_confirm" / run / f"faith_{m}" / "faithfulness_test.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("method") != m:
                problems.append(f"E1/{run}: faith_{m} holds method {d.get('method')!r}")
            else:
                present += 1
    return {"stage": "E1 explanation re-scores", "present": present, "planned": len(runs) * len(METHODS), "problems": problems}


def shelve_local_rescores(confirm_dir: Path) -> int:
    """Rename earlier local `faith_<method>` folders to `local_faith_<method>`.

    PLAN_COMPLETION.txt (addendum b) runs every E1 re-score on one machine. The 28 done locally
    before the move stay on disk as a record, but must not be mixed into the Colab-only analysis,
    which would happen silently if a Colab pair were missing and the local one still sat there.
    """
    n = 0
    for faith in sorted(confirm_dir.glob("*/faith_*")):
        target = faith.with_name("local_" + faith.name)
        if not target.exists():
            faith.rename(target)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zip", required=True)
    p.add_argument("--pinned-commit", default=None, help="default: the commit recorded in the zip's provenance.json")
    p.add_argument("--tmp", default="D:/tmp/colab_import")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    tmp = Path(args.tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(tmp)
    prov = json.loads((tmp / "provenance.json").read_text(encoding="utf-8")) if (tmp / "provenance.json").exists() else {}
    pinned = args.pinned_commit or prov.get("pinned_commit") or ""
    print(f"pinned commit: {pinned[:10] or '(none recorded)'}   zip created: {prov.get('created_utc', '?')}")

    reports = [validate_stage(ROOT, tmp, s, spec, pinned) for s, spec in STAGES.items()]
    reports.append(validate_e1(tmp, ROOT / "results"))
    bad = 0
    for r in reports:
        print(f"\n{r['stage']}: {r['present']} of {r['planned']} runs present"
              + (f"   GPU {', '.join(r['gpus'])}  torch {', '.join(r['torch'])}" if r.get("gpus") else ""))
        for line in r["problems"][:12]:
            print("  PROBLEM", line)
        if len(r["problems"]) > 12:
            print(f"  ... and {len(r['problems']) - 12} more")
        bad += len(r["problems"])
    if bad:
        print(f"\nREFUSED: {bad} problem(s). Nothing was imported.")
        return 1
    if args.dry_run:
        print("\nAll present runs passed (dry run, nothing copied).")
        return 0

    for stage in STAGES:
        src = tmp / "results" / stage
        if src.exists():
            shutil.copytree(src, ROOT / "results" / stage, dirs_exist_ok=True)
    src = tmp / "results" / "combined_confirm"
    if src.exists():
        shelved = shelve_local_rescores(ROOT / "results" / "combined_confirm")
        if shelved:
            print(f"\nShelved {shelved} earlier local re-score folder(s) as local_faith_* (kept as a record, "
                  "not analysed): E1 uses Colab re-scores only.")
        for run in src.iterdir():
            for faith in run.glob("faith_*"):
                shutil.copytree(faith, ROOT / "results" / "combined_confirm" / run.name / faith.name, dirs_exist_ok=True)
    mr = tmp / "results" / "analysis_rq3" / "matched_recall.csv"
    if mr.exists():
        (ROOT / "results" / "analysis" / "rq3").mkdir(parents=True, exist_ok=True)
        shutil.copy2(mr, ROOT / "results" / "analysis" / "rq3" / "matched_recall.csv")
    record = {"imported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pinned_commit": pinned,
              "stages": [{k: v for k, v in r.items() if k != "problems"} for r in reports]}
    (ROOT / "results" / "colab_provenance.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print("\nAll present runs passed. Imported, and recorded results/colab_provenance.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
