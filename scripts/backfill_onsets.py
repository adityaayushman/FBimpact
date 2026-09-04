"""Add annotated fall-onset frames to an existing skeleton cache.

    python scripts/backfill_onsets.py --cache data/cache/urfd --labels d:/tmp/urfd
    python scripts/backfill_onsets.py --cache data/cache/le2i

`onset_frame` was added to `ClipRecord` after both caches were built. Onset comes
from the dataset annotations, not from the skeletons, so it can be filled in
without re-running pose estimation - which would otherwise mean reprocessing
14 GB of video to recover a number already sitting in a CSV.

UR Fall: the first frame labelled transitional, from `urfall-cam0-*.csv`.
Le2i:    the annotated fall start, already carried in the cached clip's `meta`.

Idempotent: a clip that already has an onset is left alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import ClipRecord, load_cache  # noqa: E402


def urfd_onsets(labels_dir: Path) -> dict[str, int]:
    """`{clip_id: onset_frame}` from the UR Fall label CSVs."""
    from data.urfd import onset_frame, read_labels

    labels: dict[str, dict[int, int]] = {}
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        path = labels_dir / name
        if path.exists():
            labels.update(read_labels(path))

    out: dict[str, int] = {}
    for sequence, frames in labels.items():
        onset = onset_frame(frames)
        if onset is not None:
            out[f"urfd_{sequence.replace('-', '')}"] = onset
    return out


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", default="d:/tmp/urfd",
                        help="directory holding the UR Fall label CSVs")
    args = parser.parse_args(argv)

    cache = root / args.cache if not Path(args.cache).is_absolute() else Path(args.cache)
    clips = load_cache(cache)
    source = clips[0].source if clips else "unknown"

    lookup: dict[str, int] = {}
    if source == "urfd":
        lookup = urfd_onsets(Path(args.labels))
        if not lookup:
            print(f"no UR Fall label CSVs under {args.labels}")
            return 1

    updated = already = missing = invalid = 0

    for clip in clips:
        if not clip.is_fall:
            continue
        if clip.onset_frame is not None:
            already += 1
            continue

        if source == "urfd":
            onset = lookup.get(clip.clip_id)
        else:
            onset = clip.meta.get("fall_start")

        if onset is None:
            missing += 1
            continue

        onset = int(onset)
        # An onset at or after impact would make the falling phase empty or
        # negative; the annotation is unusable for this clip rather than
        # something to clamp into shape.
        if not 0 <= onset < clip.num_frames or (
            clip.impact_frame is not None and onset > clip.impact_frame
        ):
            invalid += 1
            print(f"  {clip.clip_id}: onset {onset} invalid against "
                  f"impact {clip.impact_frame} / {clip.num_frames} frames — skipped")
            continue

        clip.onset_frame = onset
        clip.meta["onset_from"] = (
            "first transitional annotation" if source == "urfd" else "annotated fall start"
        )
        clip.save(cache)
        updated += 1

    falls = sum(1 for c in clips if c.is_fall)
    print(f"\n{source}: {updated} updated, {already} already had one, "
          f"{missing} without an annotation, {invalid} invalid — of {falls} falls")

    if updated or already:
        fresh = [c for c in load_cache(cache) if c.is_fall and c.onset_frame is not None]
        if fresh:
            gaps = [c.impact_frame - c.onset_frame for c in fresh]
            fps = fresh[0].fps
            print(f"onset → impact: min {min(gaps)}  median {sorted(gaps)[len(gaps)//2]}  "
                  f"max {max(gaps)} frames "
                  f"({sorted(gaps)[len(gaps)//2] / fps:.2f}s at {fps:g} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
