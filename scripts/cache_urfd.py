"""Extract skeletons from the UR Fall zips into the cache. Stage A, real data.

    python scripts/cache_urfd.py --root d:/tmp/urfd --out data/cache/urfd

Frames are decoded from the zips in memory, posed with a frozen YOLO-Pose model,
and dropped. Nothing is unpacked to disk and no image is written anywhere - the
privacy boundary holds for real data exactly as it does for the synthetic
fixture, and the 4.2 GB of PNGs never becomes 4.2 GB of extracted files.

Impact frames come from the dataset's own per-frame annotation, never from the
skeleton; see `data/urfd.py` for why that distinction decides whether lead time
means anything.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import CACHE_SUFFIX, ClipRecord, summarise  # noqa: E402
from data.skeleton import NUM_JOINTS  # noqa: E402
from data.urfd import (  # noqa: E402
    build_clip_metadata, iter_zip_frames, list_zip_images, read_labels,
)
from pose.base import empty_keypoints, select_subject, track_greedy  # noqa: E402


def extract_sequence(zip_path: Path, estimator, subject_policy: str) -> np.ndarray:
    """Pose every frame of one sequence -> `[T, 17, 3]`."""
    frames: list[np.ndarray] = []
    previous = None
    for _, image in iter_zip_frames(zip_path):
        detections = estimator.detect(image)
        chosen = (
            track_greedy(previous, detections)
            if previous is not None
            else select_subject(detections, image.shape[:2], subject_policy)
        )
        frames.append(chosen.keypoints if chosen is not None else empty_keypoints())
        previous = chosen
        # `image` goes out of scope here; nothing is retained or written.
    if not frames:
        raise ValueError(f"no frames decoded from {zip_path.name}")
    return np.stack(frames).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="d:/tmp/urfd", help="download directory")
    parser.add_argument("--out", default="data/cache/urfd")
    parser.add_argument("--pose", default="yolo", choices=["yolo", "rtmpose"])
    parser.add_argument("--weights", default="yolo11m-pose.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subject-policy", default="largest",
                        choices=["largest", "central", "most_confident"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    source = Path(args.root)
    out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = {}
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        path = source / name
        if not path.exists():
            print(f"missing {path}. Run scripts/download_urfd.py first.")
            return 1
        labels.update(read_labels(path))
    print(f"labels for {len(labels)} sequences")

    zips = sorted((source / "zips").glob("*-cam0-rgb.zip"))
    if args.limit:
        zips = zips[: args.limit]
    if not zips:
        print(f"no sequence zips in {source / 'zips'}. Run scripts/download_urfd.py first.")
        return 1

    from pose.backends import build_estimator

    kwargs = {"device": args.device}
    if args.pose == "yolo":
        kwargs["weights"] = args.weights
    estimator = build_estimator(args.pose, **kwargs)
    print(f"pose backend: {estimator.name} (frozen)\n")

    done = skipped = failed = 0
    started = time.perf_counter()

    try:
        for i, zip_path in enumerate(zips, 1):
            sequence = zip_path.name.replace("-cam0-rgb.zip", "")
            meta = build_clip_metadata(sequence, len(list_zip_images(zip_path)), labels.get(sequence))
            target = out / f"{meta['clip_id']}{CACHE_SUFFIX}"

            if target.exists() and not args.overwrite:
                skipped += 1
                print(f"[{i:>2}/{len(zips)}] {sequence:<10} cached")
                continue

            try:
                t0 = time.perf_counter()
                keypoints = extract_sequence(zip_path, estimator, args.subject_policy)
                if keypoints.shape[1] != NUM_JOINTS:
                    raise ValueError(f"estimator returned {keypoints.shape[1]} joints")

                # A fall sequence whose annotation never reaches "lying" has no
                # usable t*; recorded as ADL rather than guessed at.
                impact = meta["impact_frame"]
                if impact is not None:
                    impact = min(int(impact), keypoints.shape[0] - 1)

                ClipRecord(
                    clip_id=meta["clip_id"], subject=meta["subject"], keypoints=keypoints,
                    fps=meta["fps"], label=meta["label"], impact_frame=impact,
                    activity=meta["activity"], view=meta["view"], source=meta["source"],
                    meta=meta["meta"],
                ).save(out)

                done += 1
                coverage = float((keypoints[..., 2] >= 0.3).mean())
                elapsed = time.perf_counter() - t0
                marker = f"t*={impact}" if impact is not None else "no impact"
                print(f"[{i:>2}/{len(zips)}] {sequence:<10} {keypoints.shape[0]:>4}f  "
                      f"{marker:<12} joints {coverage:5.1%}  {elapsed:5.1f}s", flush=True)
            except Exception:
                failed += 1
                print(f"[{i:>2}/{len(zips)}] {sequence:<10} FAILED")
                traceback.print_exc(limit=2)
    finally:
        estimator.close()

    print(f"\n{done} extracted, {skipped} cached, {failed} failed "
          f"in {time.perf_counter() - started:.0f}s -> {out}")

    from data.clips import load_cache
    stats = summarise(load_cache(out))
    print(f"cache: {stats['clips']} clips | {stats['falls']} falls | {stats['adls']} ADL | "
          f"{stats['total_hours'] * 60:.1f} min")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
