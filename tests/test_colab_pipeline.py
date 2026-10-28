"""The Colab hand-off: merging worker folders, and the gate that refuses wrong or mismatched runs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import import_colab_results as ic  # noqa: E402
import make_colab_bundle as mb  # noqa: E402
import merge_worker_dirs as mw  # noqa: E402

COMMIT = "a" * 40


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------ merge
def make_worker(workers: Path, wname: str, seed: int, folds=(0, 1)):
    rows = []
    for fold in folds:
        run = f"cfg_seed{seed}_2026{fold}"
        d = workers / wname / run
        d.mkdir(parents=True)
        (d / "x.txt").write_text("hi")
        rows.append({"config": "cfg", "seed": seed, "fold": fold, "run_dir": f"results\\workers\\{wname}\\{run}", "recall": 0.5})
    write_csv(workers / wname / "runs.csv", rows)


def test_merge_moves_runs_and_rewrites_paths_to_portable_form(tmp_path):
    workers = tmp_path / "w"
    make_worker(workers, "seed1", 1)
    make_worker(workers, "seed2", 2)
    out = tmp_path / "merged"
    r = mw.merge(workers, out, "merged")
    assert r["runs"] == 4 and r["moved"] == 4
    rows = list(csv.DictReader((out / "runs.csv").open(encoding="utf-8")))
    assert all(x["run_dir"].startswith("results/merged/") and "\\" not in x["run_dir"] for x in rows)
    assert all((out / Path(x["run_dir"]).name / "x.txt").exists() for x in rows)


def test_merge_is_idempotent_and_keeps_adding(tmp_path):
    workers, out = tmp_path / "w", tmp_path / "merged"
    make_worker(workers, "seed1", 1)
    mw.merge(workers, out, "merged")
    again = mw.merge(workers, out, "merged")
    assert again["runs"] == 2 and again["moved"] == 0
    make_worker(workers, "seed2", 2)
    assert mw.merge(workers, out, "merged")["runs"] == 4


def test_a_row_without_a_run_folder_is_not_kept(tmp_path):
    workers, out = tmp_path / "w", tmp_path / "merged"
    make_worker(workers, "seed1", 1, folds=(0,))
    rows = list(csv.DictReader((workers / "seed1" / "runs.csv").open(encoding="utf-8")))
    rows.append({"config": "cfg", "seed": 1, "fold": 3, "run_dir": "results\\workers\\seed1\\never_finished", "recall": 0.1})
    write_csv(workers / "seed1" / "runs.csv", rows)
    assert mw.merge(workers, out, "merged")["runs"] == 1


# ------------------------------------------------------------------------------ the import gate
def make_run(root: Path, stage: str, variant: str, seed: int, fold: int, *, cfg_edit=None, env_edit=None, bench="urfd"):
    cfg = yaml.safe_load((ROOT / "configs" / "frozen" / bench / f"{variant}.yaml").read_text(encoding="utf-8"))
    cfg["run"]["seed"] = seed                      # legitimately differs per run
    cfg["data"]["split"]["fold"] = fold
    if cfg_edit:
        cfg_edit(cfg)
    run = f"{variant}_seed{seed}_f{fold}"
    d = root / "results" / stage / run
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {"cuda_available": True, "cuda_device": "Tesla T4", "torch": "2.5.1+cu121", "git_commit": COMMIT}
    if env_edit:
        env_edit(env)
    (d / "environment.json").write_text(json.dumps(env), encoding="utf-8")
    (d / "outcomes_test.csv").write_text("clip_id\n", encoding="utf-8")
    csv_path = root / "results" / stage / "runs.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8"))) if csv_path.exists() else []
    rows.append({"config": variant, "seed": seed, "fold": fold, "run_dir": f"results/{stage}/{run}"})
    write_csv(csv_path, rows)


def stage_report(tmp_path):
    return ic.validate_stage(ROOT, tmp_path, "urfd_colab", ic.STAGES["urfd_colab"], COMMIT)


def test_a_correct_run_passes_and_missing_runs_are_reported_not_hidden(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0)
    r = stage_report(tmp_path)
    assert r["problems"] == [] and r["present"] == 1 and r["planned"] == 90 and len(r["missing"]) == 89


def test_a_changed_training_setting_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0, cfg_edit=lambda c: c["train"].__setitem__("epochs", 5))
    r = stage_report(tmp_path)
    assert any("train.epochs" in p for p in r["problems"])


def test_the_wrong_ablation_switch_is_refused(tmp_path):
    # a run recorded as no_velocity whose config still has velocity on
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0, cfg_edit=lambda c: c["features"].__setitem__("with_velocity", True))
    assert any("with_velocity" in p for p in stage_report(tmp_path)["problems"])


def test_a_run_from_another_code_version_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0, env_edit=lambda e: e.__setitem__("git_commit", "b" * 40))
    assert any("pinned" in p for p in stage_report(tmp_path)["problems"])


def test_a_run_without_cuda_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0, env_edit=lambda e: e.__setitem__("cuda_available", False))
    assert any("without CUDA" in p for p in stage_report(tmp_path)["problems"])


def test_a_run_forced_onto_the_cpu_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 0, cfg_edit=lambda c: c["run"].__setitem__("device", "cpu"))
    assert any("run.device" in p for p in stage_report(tmp_path)["problems"])


def test_a_run_trained_on_another_fold_than_recorded_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 1, 2, cfg_edit=lambda c: c["data"]["split"].__setitem__("fold", 4))
    assert any("trained on fold 4, recorded as fold 2" in p for p in stage_report(tmp_path)["problems"])


def test_earlier_local_rescores_are_shelved_so_they_cannot_mix_into_the_colab_analysis(tmp_path):
    run = tmp_path / "combined_confirm" / "combined_seed1_x"
    (run / "faith_gradient").mkdir(parents=True)
    (run / "faith_gradient" / "faithfulness_test.json").write_text("{}")
    assert ic.shelve_local_rescores(tmp_path / "combined_confirm") == 1
    assert not (run / "faith_gradient").exists() and (run / "local_faith_gradient" / "faithfulness_test.json").exists()
    assert ic.shelve_local_rescores(tmp_path / "combined_confirm") == 0          # idempotent


def test_a_run_outside_the_plan_is_refused(tmp_path):
    make_run(tmp_path, "urfd_colab", "no_velocity", 7, 0)        # seed 7 is not planned
    assert any("not part of the plan" in p for p in stage_report(tmp_path)["problems"])


def test_the_rq3_expected_configs_differ_only_in_the_loss_weight():
    b = ic.load_expected(ROOT, "configs/combined.yaml", ic.CONFIRM_OVERRIDES)
    g = ic.load_expected(ROOT, "configs/combined_preimpact.yaml", ic.CONFIRM_OVERRIDES)
    assert b["train"]["epochs"] == 30 and b["train"]["batch_size"] == 24
    assert ic.diff(ic.relevant(b), ic.relevant(g)) == ["loss.lam: expected 0.0, got 1.5"]


def test_the_bundle_builder_records_a_checksum_for_every_file(tmp_path):
    cache = tmp_path / "data" / "cache" / "urfd"
    cache.mkdir(parents=True)
    (cache / "a.npz").write_bytes(b"123")
    old = mb.CACHES
    mb.CACHES = ["data/cache/urfd"]
    try:
        z = mb.build_data_bundle(tmp_path, tmp_path / "out")
    finally:
        mb.CACHES = old
    import zipfile
    with zipfile.ZipFile(z) as zf:
        m = json.loads(zf.read("MANIFEST.json"))
    assert list(m["files"]) == ["data/cache/urfd/a.npz"] and len(m["files"]["data/cache/urfd/a.npz"]["sha256"]) == 64
