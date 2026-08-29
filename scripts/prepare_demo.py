"""Bundle a checkpoint and a handful of demo clips for the deployed API.

    python scripts/prepare_demo.py --checkpoint results/smoke_seed0/best.pt

The clips are chosen to make the demo honest rather than flattering: three falls
of different directions, and three **hard negatives** - sitting down, bending to
pick something up, lying down. All three of those are controlled descents, and a
demo that only showed walking-versus-falling would hide the failure mode that
actually matters.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import load_cache  # noqa: E402

# (activity, how many) - falls first, then the hard negatives.
DEMO_SELECTION = [
    ("fall_forward", 1),
    ("fall_backward", 1),
    ("fall_sideways", 1),
    ("sit_down", 1),
    ("pick_up", 1),
    ("lie_down", 1),
    ("walk", 1),
]


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="results/smoke_seed0/best.pt")
    parser.add_argument("--cache", default="data/cache/synthetic")
    parser.add_argument("--out", default="backend")
    args = parser.parse_args(argv)

    out = root / args.out
    model_dir = out / "model"
    clips_dir = out / "demo_clips"
    model_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = root / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(
            f"no checkpoint at {checkpoint}. Train one first:\n"
            f"  python train.py --config configs/ours_preimpact.yaml"
        )
    shutil.copy2(checkpoint, model_dir / "best.pt")
    print(f"checkpoint -> {model_dir / 'best.pt'} "
          f"({(model_dir / 'best.pt').stat().st_size / 1e6:.1f} MB)")

    for existing in clips_dir.glob("*.npz"):
        existing.unlink()

    clips = load_cache(root / args.cache)
    # Prefer clips from a single subject so the demo set is self-consistent, and
    # fall back to any subject when an activity is missing from that one.
    by_activity: dict[str, list] = {}
    for clip in sorted(clips, key=lambda c: c.clip_id):
        by_activity.setdefault(clip.activity, []).append(clip)

    written = 0
    for activity, count in DEMO_SELECTION:
        available = by_activity.get(activity, [])
        if not available:
            print(f"  (skipped {activity}: none in the cache)")
            continue
        for clip in available[:count]:
            clip.save(clips_dir)
            marker = f"t*={clip.impact_frame}" if clip.is_fall else "no impact"
            print(f"  {clip.clip_id:<20} {activity:<16} {clip.num_frames:>4}f  {marker}")
            written += 1

    total = sum(p.stat().st_size for p in clips_dir.glob("*.npz"))
    print(f"\n{written} demo clips -> {clips_dir} ({total / 1024:.0f} kB)")
    print("\nnext: uvicorn backend.app:app --reload --port 8000")


if __name__ == "__main__":
    main()
