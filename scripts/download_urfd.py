"""Fetch the UR Fall Detection dataset (Kwolek & Kepski).

    python scripts/download_urfd.py --out d:/tmp/urfd

30 fall sequences and 40 activity-of-daily-living sequences, RGB from camera 0,
plus the per-frame label CSVs. About 4.2 GB.

The label CSVs are the reason this dataset is usable here at all. UR Fall ships
a per-frame annotation - `-1` upright, `0` transitional, `1` lying - which gives
a principled impact frame without asking a human to re-annotate 30 clips, and
without deriving `t*` from the skeleton the model consumes, which would make
lead time partly circular.

Downloads are skipped if the file already exists with the right size, so an
interrupted run can simply be repeated.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://fenix.ur.edu.pl/mkepski/ds/data"

LABEL_FILES = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
NUM_FALLS = 30
NUM_ADLS = 40


def sequence_names() -> list[str]:
    return (
        [f"fall-{i:02d}" for i in range(1, NUM_FALLS + 1)]
        + [f"adl-{i:02d}" for i in range(1, NUM_ADLS + 1)]
    )


def remote_size(url: str, timeout: float = 30.0) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def download(url: str, target: Path, timeout: float = 120.0) -> tuple[bool, str]:
    """Fetch `url` to `target`, skipping a complete existing file."""
    expected = remote_size(url)
    if target.exists() and expected is not None and target.stat().st_size == expected:
        return True, "cached"
    if target.exists():
        target.unlink()

    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        return False, str(exc)

    tmp.replace(target)
    return True, f"{target.stat().st_size / 1e6:.1f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="d:/tmp/urfd", help="download directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N sequences, for a quick trial")
    args = parser.parse_args(argv)

    out = Path(args.out)
    zips = out / "zips"
    zips.mkdir(parents=True, exist_ok=True)

    for name in LABEL_FILES:
        ok, note = download(f"{BASE}/{name}", out / name)
        print(f"  labels {name:<26} {'ok' if ok else 'FAILED'}  {note}", flush=True)
        if not ok:
            print("  cannot continue without the label CSVs", flush=True)
            return 1

    names = sequence_names()[: args.limit] if args.limit else sequence_names()
    failed: list[str] = []

    for i, name in enumerate(names, 1):
        filename = f"{name}-cam0-rgb.zip"
        ok, note = download(f"{BASE}/{filename}", zips / filename)
        status = "ok" if ok else "FAILED"
        print(f"[{i:>2}/{len(names)}] {filename:<24} {status:<7} {note}", flush=True)
        if not ok:
            failed.append(name)

    total = sum(p.stat().st_size for p in zips.glob("*.zip"))
    print(f"\n{len(names) - len(failed)}/{len(names)} sequences, {total / 1e9:.2f} GB in {zips}")
    if failed:
        print(f"failed: {', '.join(failed)} — re-run to retry just those")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
