"""Bundle a checkpoint and a handful of demo clips for the deployed API.

    python scripts/prepare_demo.py --checkpoint results/urfd/<run>/best.pt \
                                   --cache data/cache/urfd

The clips are chosen to make the demo honest rather than flattering. On the
synthetic fixture that means three falls of different directions plus the
controlled descents - sitting, bending, lying down - because a demo of
walking-versus-falling would hide the failure mode that matters.

On UR Fall the same principle picks differently: the ADL sequences containing a
**deliberate lie-down** are the hard negatives, because "the person is now
horizontal" is exactly the shortcut a fall detector is tempted to learn. Those
are preferred over sequences where the subject stays upright throughout.

Clips are drawn from the checkpoint's **test** fold wherever the checkpoint
records one. Demonstrating a model on sequences it trained on is not a demo, it
is a memorisation check, and it would show lead times the model could not
reproduce on anything it had not seen.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import ClipRecord, load_cache  # noqa: E402

# Synthetic fixture: (activity, how many). Falls first, then hard negatives.
SYNTHETIC_SELECTION = [
    ("fall_forward", 1), ("fall_backward", 1), ("fall_sideways", 1),
    ("sit_down", 1), ("pick_up", 1), ("lie_down", 1), ("walk", 1),
]

N_REAL_FALLS = 4
N_REAL_ADLS = 4


def held_out_subjects(checkpoint: Path) -> set[str] | None:
    """Test-split subjects recorded in the checkpoint, if it has any."""
    try:
        import torch

        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    split = blob.get("split") or {}
    test = split.get("test")
    return set(test) if test else None


def pick_synthetic(clips: list[ClipRecord]) -> list[ClipRecord]:
    by_activity: dict[str, list[ClipRecord]] = {}
    for clip in sorted(clips, key=lambda c: c.clip_id):
        by_activity.setdefault(clip.activity, []).append(clip)

    chosen: list[ClipRecord] = []
    for activity, count in SYNTHETIC_SELECTION:
        available = by_activity.get(activity, [])
        if not available:
            print(f"  (skipped {activity}: none in the cache)")
            continue
        chosen.extend(available[:count])
    return chosen


def pick_real(clips: list[ClipRecord]) -> list[ClipRecord]:
    """Falls with the most pre-impact room, and ADLs that involve lying down."""
    falls = [c for c in clips if c.is_fall]
    adls = [c for c in clips if not c.is_fall]

    # A fall with more frames before t* gives the model a chance to anticipate
    # rather than only to react, which is what the demo is showing.
    falls.sort(key=lambda c: -(c.impact_frame or 0))

    # `lying_frames` is recorded by the UR Fall adapter: an ADL sequence with
    # many of them is a deliberate lie-down, the hardest negative available.
    adls.sort(key=lambda c: -(c.meta.get("lying_frames", 0) or 0))

    hard = [c for c in adls if (c.meta.get("lying_frames", 0) or 0) > 0][: N_REAL_ADLS - 1]
    upright = [c for c in adls if (c.meta.get("lying_frames", 0) or 0) == 0][:1]
    return falls[:N_REAL_FALLS] + hard + upright


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="data/cache/urfd")
    parser.add_argument("--out", default="backend")
    parser.add_argument("--all-folds", action="store_true",
                        help="ignore the checkpoint's test split and pick from the whole cache")
    args = parser.parse_args(argv)

    out = root / args.out
    model_dir = out / "model"
    clips_dir = out / "demo_clips"
    model_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = root / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    shutil.copy2(checkpoint, model_dir / "best.pt")
    print(f"checkpoint -> {model_dir / 'best.pt'} "
          f"({(model_dir / 'best.pt').stat().st_size / 1e6:.1f} MB)")

    cache = root / args.cache if not Path(args.cache).is_absolute() else Path(args.cache)
    clips = load_cache(cache)
    source = clips[0].source if clips else "unknown"

    test_subjects = None if args.all_folds else held_out_subjects(checkpoint)
    if test_subjects:
        held = [c for c in clips if c.subject in test_subjects]
        falls_held = sum(1 for c in held if c.is_fall)
        if falls_held >= 2:
            clips = held
            print(f"restricted to the checkpoint's {len(test_subjects)} held-out groups "
                  f"({len(clips)} clips, {falls_held} falls) — unseen during training")
        else:
            print(f"WARNING: the held-out fold has only {falls_held} fall(s); "
                  f"picking from the whole cache instead. These clips were SEEN in training.")

    chosen = pick_real(clips) if source != "synthetic" else pick_synthetic(clips)
    if not chosen:
        raise SystemExit(f"no clips selected from {cache}")

    for existing in clips_dir.glob("*.npz"):
        existing.unlink()

    for clip in chosen:
        # UR Fall records activity as just "fall"/"adl" and publishes no
        # description of what each subject actually did, so the sequence id is
        # the most specific honest label available for the clip picker.
        sequence = clip.meta.get("sequence")
        if sequence and clip.activity in ("fall", "adl"):
            lying = clip.meta.get("lying_frames", 0) or 0
            clip.activity = (
                sequence if clip.is_fall or lying == 0 else f"{sequence} (lies down)"
            )
        clip.save(clips_dir)
        marker = f"t*={clip.impact_frame}" if clip.is_fall else (
            f"lying={clip.meta.get('lying_frames', 0)}" if clip.meta.get("lying_frames") else "upright"
        )
        print(f"  {clip.clip_id:<22} {clip.activity:<14} {clip.num_frames:>4}f  {marker}")

    total = sum(p.stat().st_size for p in clips_dir.glob("*.npz"))
    print(f"\n{len(chosen)} demo clips from '{source}' -> {clips_dir} ({total / 1024:.0f} kB)")
    print("\nnext: uvicorn backend.app:app --reload --port 8000")


if __name__ == "__main__":
    main()
