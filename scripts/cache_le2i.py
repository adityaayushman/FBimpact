"""Extract skeletons from the Le2i archive into the cache. Stage A, real data.

    python scripts/cache_le2i.py --archive d:/tmp/le2i/falldataset-imvia.zip \
                                --out data/cache/le2i

Only the Home and Coffee room scenes are processed: they are the only ones with
annotation files, and without an annotated impact frame there is no lead time to
measure.

Unlike UR Fall, whose sequences are PNG frames that decode straight from memory,
Le2i ships AVI files - and a video decoder needs a real seekable file. Each clip
is therefore written to a scratch file, decoded, posed and **deleted before the
next one starts**, so at most one video exists on disk at a time and no frame is
ever written anywhere. The privacy boundary is preserved; the difference is only
in how the bytes reach the decoder.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import CACHE_SUFFIX, ClipRecord, summarise  # noqa: E402
from data.le2i import build_clip_metadata, clip_key, parse_annotation, scene_of  # noqa: E402
from data.skeleton import NUM_JOINTS  # noqa: E402
from pose.base import empty_keypoints, select_subject, track_greedy  # noqa: E402

VIDEO_SUFFIXES = (".avi", ".mp4", ".mov")


def index_archive(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    """Map `clip_key -> {"video": name, "annotation": name}` for annotated scenes."""
    index: dict[str, dict[str, str]] = {}
    for name in archive.namelist():
        if name.endswith("/") or scene_of(name) is None:
            continue
        key = clip_key(name)
        if key is None:
            continue
        lower = name.lower()
        if lower.endswith(VIDEO_SUFFIXES):
            index.setdefault(key, {})["video"] = name
        elif lower.endswith(".txt"):
            index.setdefault(key, {})["annotation"] = name
    return index


def pose_video(path: Path, estimator, subject_policy: str) -> tuple[np.ndarray, float]:
    """Decode a video file and pose every frame -> `([T, 17, 3], fps)`."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"could not open {path.name}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        frames: list[np.ndarray] = []
        previous = None
        while True:
            ok, image = capture.read()
            if not ok:
                break
            detections = estimator.detect(image)
            chosen = (
                track_greedy(previous, detections)
                if previous is not None
                else select_subject(detections, image.shape[:2], subject_policy)
            )
            frames.append(chosen.keypoints if chosen is not None else empty_keypoints())
            previous = chosen
            # `image` goes out of scope here; nothing is retained or written.
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"no frames decoded from {path.name}")
    return np.stack(frames).astype(np.float32), fps


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", default="d:/tmp/le2i/falldataset-imvia.zip")
    parser.add_argument("--out", default="data/cache/le2i")
    parser.add_argument("--scratch", default="d:/tmp/le2i/scratch",
                        help="where a single video is unpacked at a time")
    parser.add_argument("--pose", default="yolo")
    parser.add_argument("--weights", default="yolo11s-pose.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subject-policy", default="largest",
                        choices=["largest", "central", "most_confident"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"missing {archive_path}. Run scripts/download_le2i.py first.")
        return 1

    out = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    from pose.backends import build_estimator

    kwargs = {"device": args.device}
    if args.pose == "yolo":
        kwargs["weights"] = args.weights
    estimator = build_estimator(args.pose, **kwargs)
    print(f"pose backend: {estimator.name} (frozen)")

    done = skipped = failed = no_annotation = 0
    started = time.perf_counter()

    with zipfile.ZipFile(archive_path) as archive:
        index = index_archive(archive)
        keys = sorted(k for k, v in index.items() if "video" in v)
        if args.limit:
            keys = keys[: args.limit]
        print(f"{len(keys)} annotated-scene clips in the archive\n")

        try:
            for i, key in enumerate(keys, 1):
                entry = index[key]
                target = out / f"le2i_{key}{CACHE_SUFFIX}"
                if target.exists() and not args.overwrite:
                    skipped += 1
                    print(f"[{i:>3}/{len(keys)}] {key:<26} cached")
                    continue

                if "annotation" not in entry:
                    no_annotation += 1
                    print(f"[{i:>3}/{len(keys)}] {key:<26} no annotation — skipped")
                    continue

                temp = None
                try:
                    annotation = parse_annotation(
                        archive.read(entry["annotation"]).decode("utf-8", "replace")
                    )

                    suffix = Path(entry["video"]).suffix or ".avi"
                    with tempfile.NamedTemporaryFile(
                        dir=scratch, suffix=suffix, delete=False
                    ) as handle:
                        handle.write(archive.read(entry["video"]))
                        temp = Path(handle.name)

                    t0 = time.perf_counter()
                    keypoints, fps = pose_video(temp, estimator, args.subject_policy)
                    if keypoints.shape[1] != NUM_JOINTS:
                        raise ValueError(f"estimator returned {keypoints.shape[1]} joints")

                    meta = build_clip_metadata(key, annotation, keypoints.shape[0])
                    ClipRecord(
                        clip_id=meta["clip_id"], subject=meta["subject"], keypoints=keypoints,
                        # Trust the container's frame rate when it looks sane,
                        # otherwise fall back to the dataset's documented 25 fps.
                        fps=fps if 5.0 < fps < 120.0 else meta["fps"],
                        label=meta["label"], impact_frame=meta["impact_frame"],
                        activity=meta["activity"], view=meta["view"], source=meta["source"],
                        meta=meta["meta"],
                    ).save(out)

                    done += 1
                    coverage = float((keypoints[..., 2] >= 0.3).mean())
                    marker = (f"t*={meta['impact_frame']}" if meta["impact_frame"] is not None
                              else "no fall")
                    print(f"[{i:>3}/{len(keys)}] {key:<26} {keypoints.shape[0]:>4}f  "
                          f"{marker:<12} joints {coverage:5.1%}  "
                          f"{time.perf_counter() - t0:5.1f}s", flush=True)
                except Exception:
                    failed += 1
                    print(f"[{i:>3}/{len(keys)}] {key:<26} FAILED")
                    traceback.print_exc(limit=2)
                finally:
                    # Deleted before the next clip is unpacked, so at most one
                    # video exists on disk at any moment.
                    if temp is not None:
                        temp.unlink(missing_ok=True)
        finally:
            estimator.close()

    print(f"\n{done} extracted, {skipped} cached, {no_annotation} unannotated, "
          f"{failed} failed in {time.perf_counter() - started:.0f}s -> {out}")

    from data.clips import load_cache

    try:
        stats = summarise(load_cache(out))
        print(f"cache: {stats['clips']} clips | {stats['falls']} falls | "
              f"{stats['adls']} ADL | {stats['total_hours'] * 60:.1f} min")
    except FileNotFoundError:
        print("cache is empty")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
