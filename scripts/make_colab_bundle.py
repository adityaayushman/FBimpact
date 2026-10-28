"""Build the two files you upload to Google Drive for the Colab run (PLAN_COMPLETION.txt).

    python scripts/make_colab_bundle.py

Writes to `colab/` (git-ignored, because it holds derived skeleton data):

  fbimpact_data_bundle.zip   the skeleton caches (UR Fall, Le2i) + MANIFEST.json with a sha256 of
                             every file, so the notebook can prove Colab trains on exactly this data.
  fbimpact_checkpoints.zip   the 40 saved confirmation-grid checkpoints (arms B and C) + a runs.csv
                             for them, for E1 (re-scoring with every explanation method).

No torch. The skeletons are derived from the public UR Fall and Le2i datasets; they are not committed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"
CACHES = ["data/cache/urfd", "data/cache/le2i"]
CHECKPOINT_ARMS = ("combined", "combined_phases")
KEEP = ("best.pt", "config.yaml", "split.json", "environment.json", "report_test.json", "training_summary.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_data_bundle(root: Path = ROOT, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for cache in CACHES:
        for p in sorted((root / cache).glob("*.npz")):
            files[p.relative_to(root).as_posix()] = p
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {rel: {"sha256": sha256(p), "bytes": p.stat().st_size} for rel, p in files.items()},
    }
    path = out / "fbimpact_data_bundle.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, p in files.items():
            z.write(p, rel)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
    return path


def build_checkpoint_bundle(root: Path = ROOT, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    src = root / "results" / "combined_confirm"
    with (src / "runs.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["config"] in CHECKPOINT_ARMS]
    fields = list(rows[0].keys())
    kept = []
    path = out / "fbimpact_checkpoints.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            run = Path(r["run_dir"].replace("\\", "/")).name
            run_dir = src / run
            for name in KEEP:
                if (run_dir / name).exists():
                    z.write(run_dir / name, f"results/combined_confirm/{run}/{name}")
            r["run_dir"] = f"results/combined_confirm/{run}"      # relative, forward slashes: valid on Colab and Windows
            kept.append(r)
        sio = io.StringIO()
        w = csv.DictWriter(sio, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
        z.writestr("results/combined_confirm/runs.csv", sio.getvalue())
    return path


def main() -> int:
    a = build_data_bundle()
    b = build_checkpoint_bundle()
    for p in (a, b):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
