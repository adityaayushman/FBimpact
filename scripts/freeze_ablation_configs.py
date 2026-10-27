"""E2 (PLAN_COMPLETION.txt): freeze the exact configs of the existing ablation runs.

    python scripts/freeze_ablation_configs.py

The existing UR Fall / Le2i ablation runs were launched with settings that live in their saved
`config.yaml` and not in configs/ablations/ (those inherit from default.yaml). To add seeds
without drifting from them, this writes each variant's SAVED config to
`configs/frozen/<benchmark>/<name>.yaml`, named by the config stem the runs were recorded
under, so `run_ablations.py --resume` recognises what already exists.

Le2i never ran no_velocity or no_grounding. Those are built from the Le2i full-model saved
config plus the ablation's own overlay (configs/ablations/<name>.yaml, without its `_base_`
and run name), and the result is printed as a diff so the only changes are visible.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ROOT / "configs" / "ablations"
NAMES = ["ours_preimpact", "baseline_stgcn", "no_preimpact_loss", "no_temporal", "no_velocity", "no_grounding"]


def first_saved(benchmark: str, name: str) -> dict | None:
    with (ROOT / "results" / benchmark / "runs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["config"] == name:
                p = Path(row["run_dir"])
                p = p if p.is_absolute() else ROOT / p
                return yaml.safe_load((p / "config.yaml").read_text(encoding="utf-8"))
    return None


def overlay(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = overlay(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    lines = []
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            lines += diff(a[k], b[k], f"{prefix}{k}.")
        elif a.get(k) != b.get(k):
            lines.append(f"{prefix}{k}: {a.get(k)!r} -> {b.get(k)!r}")
    return lines


def main() -> int:
    for benchmark in ("urfd", "le2i"):
        out_dir = ROOT / "configs" / "frozen" / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)
        full = first_saved(benchmark, "ours_preimpact")
        assert full, f"{benchmark}: no saved ours_preimpact run"
        print(f"\n== {benchmark}: each variant against the saved full-model config")
        for name in NAMES:
            cfg = first_saved(benchmark, name)
            origin = "saved config of an existing run"
            if cfg is None:
                extra = yaml.safe_load((ABLATIONS / f"{name}.yaml").read_text(encoding="utf-8"))
                extra.pop("_base_", None)
                cfg = overlay(full, extra)
                origin = "full-model saved config + configs/ablations overlay (never run before)"
            (out_dir / f"{name}.yaml").write_text(
                f"# Frozen for PLAN_COMPLETION.txt E2: {origin}.\n" + yaml.safe_dump(cfg, sort_keys=False),
                encoding="utf-8")
            d = [line for line in diff(full, cfg) if not line.startswith(("run.name", "run.seed", "run.results_dir"))]
            print(f"  {name:20s} [{origin[:28]}...]  differs from full model in: {d if d else 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
