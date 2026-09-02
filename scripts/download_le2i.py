"""Fetch the Le2i / ImViA fall dataset from its Kaggle mirror.

    python scripts/download_le2i.py --out d:/tmp/le2i

Le2i has four scenes, but only **Home** and **Coffee room** ship annotation
files, and those give the frame at which a fall begins and ends. Only those two
scenes are usable here: without an annotation there is no impact frame, and
without an impact frame there is no lead time to measure. The Office and Lecture
room subsets are downloaded with the archive but ignored downstream.

The original le2i.cnrs.fr host is dead and the ImViA portal returns 403 to
scripted clients, so the Kaggle mirror is the practical source. It needs a
`~/.kaggle/kaggle.json` API token.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SLUG = "tuyenldvn/falldataset-imvia"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="d:/tmp/le2i")
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--unzip", action="store_true",
                        help="unpack the archive; off by default because the "
                             "pose extractor streams frames straight out of it")
    args = parser.parse_args(argv)

    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    try:
        import kaggle
    except ImportError:
        print("needs the kaggle client:  pip install kaggle")
        return 1
    except OSError as exc:
        print(f"kaggle credentials not found ({exc}).\n"
              f"Create an API token at kaggle.com/settings and save it to "
              f"~/.kaggle/kaggle.json")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = kaggle.KaggleApi()
    api.authenticate()
    print(f"downloading {args.slug} -> {out}")

    api.dataset_download_files(args.slug, path=str(out), unzip=args.unzip, quiet=False)

    archives = sorted(out.glob("*.zip"))
    total = sum(p.stat().st_size for p in archives) + sum(
        p.stat().st_size for p in out.rglob("*") if p.is_file() and p.suffix != ".zip"
    )
    print(f"\n{total / 1e9:.2f} GB in {out}")
    for a in archives:
        print(f"  {a.name}  {a.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
