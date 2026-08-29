"""Run directories, metric logs and result tables.

Deliberately dependency-free: JSONL for the per-epoch stream, JSON for the final
report, CSV for the tables that go into the paper. A reviewer with Python and
nothing else can read every artefact a run produces.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    """Make NumPy scalars, arrays and dataclasses JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def create_run_dir(root: str | Path, name: str, timestamp: bool = True) -> Path:
    """Create `results/<name>` or `results/<name>_<utc timestamp>`."""
    root = Path(root)
    if timestamp:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{name}_{stamp}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_logger(name: str, run_dir: Path | None = None, level: int = logging.INFO):
    """Console logger, mirrored to `<run_dir>/run.log` when given a run dir."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if run_dir is not None:
        file_handler = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class JsonlLogger:
    """Append-only metric stream, one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def save_csv(rows: list[dict], path: str | Path) -> Path:
    """Write a list of flat dicts as CSV, unioning keys across rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (json.dumps(v, default=_json_default) if isinstance(v, (dict, list)) else v)
                 for k, v in row.items()}
            )
    return path


def environment_report() -> dict:
    """Versions and git state, written into every run directory.

    A results table with no record of which commit produced it is not
    reproducible, however carefully the seeds were set.
    """
    import torch

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv,
    }


def aggregate_seeds(rows: list[dict], keys: list[str] | None = None) -> dict:
    """Mean and standard deviation across seeds (Section 14: never a single run)."""
    if not rows:
        return {}
    keys = keys or [
        k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out: dict[str, float] = {"n_runs": len(rows)}
    for key in keys:
        values = np.array(
            [r[key] for r in rows if isinstance(r.get(key), (int, float))], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render rows as a fixed-width table for the console and the run log."""
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0].keys())

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "nan" if not np.isfinite(value) else f"{value:.4f}"
        return str(value)

    widths = {
        c: max(len(c), *(len(cell(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    rule = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join(
        "  ".join(cell(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{header}\n{rule}\n{body}"
