"""Generate the synthetic skeleton cache used for smoke tests.

    python scripts/make_synthetic.py
    python scripts/make_synthetic.py --subjects 16 --out data/cache/synthetic_big

This is a fixture, not a benchmark. It exists so the pipeline can be run,
tested and profiled before UP-Fall is downloaded and annotated - nothing
measured on it belongs in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import summarise  # noqa: E402
from data.synthetic import make_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/cache/synthetic")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--falls", type=int, default=6, help="fall clips per subject")
    parser.add_argument("--adls", type=int, default=10, help="ADL clips per subject")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="delete existing clips first")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        for path in out_dir.glob("*.npz"):
            path.unlink()

    clips = make_dataset(
        num_subjects=args.subjects,
        falls_per_subject=args.falls,
        adls_per_subject=args.adls,
        fps=args.fps,
        seed=args.seed,
    )
    for clip in clips:
        clip.save(out_dir)

    stats = summarise(clips)
    print(f"wrote {stats['clips']} clips to {out_dir}")
    print(
        f"  {stats['falls']} falls, {stats['adls']} ADL, {stats['subjects']} subjects, "
        f"{stats['total_hours'] * 60:.1f} minutes"
    )
    return out_dir


if __name__ == "__main__":
    main()
