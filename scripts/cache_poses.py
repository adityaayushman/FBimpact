"""Stage A - extract skeletons from videos into the cache. Run once per dataset.

    python scripts/cache_poses.py --dataset upfall --root D:/data/UP-Fall --out data/cache/upfall
    python scripts/cache_poses.py --dataset generic --root D:/data/le2i --out data/cache/le2i

After this, nothing downstream touches a pixel. This is where the privacy
boundary in Section 19 is actually enforced: the cache holds joint coordinates
and confidences, and no frame, crop or thumbnail is written anywhere.

Falls are cached with a placeholder impact frame; `scripts/annotate_onsets.py`
supplies the real `t*`.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.upfall import find_videos, parse_name  # noqa: E402
from pose.backends import build_estimator  # noqa: E402
from pose.cache import cache_stats, cache_video  # noqa: E402


def cache_upfall(root: Path, out: Path, estimator, args) -> tuple[int, int]:
    videos = find_videos(root)
    print(f"found {len(videos)} UP-Fall recordings under {root}")
    done = failed = 0
    for index, path in enumerate(videos, 1):
        key = parse_name(path.name)
        try:
            cache_video(
                video_path=path,
                out_dir=out,
                estimator=estimator,
                clip_id=key.clip_id,
                subject=f"S{key.subject:02d}",
                label="fall" if key.is_fall else "adl",
                impact_frame=0 if key.is_fall else None,
                activity=key.activity_name,
                view=f"cam{key.camera}",
                source="upfall",
                overwrite=args.overwrite,
                subject_policy=args.subject_policy,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            done += 1
        except Exception:
            failed += 1
            print(f"  [{index}/{len(videos)}] FAILED {path.name}")
            traceback.print_exc(limit=1)
        if index % 25 == 0:
            print(f"  [{index}/{len(videos)}] cached {done}, failed {failed}")
    return done, failed


def cache_generic(root: Path, out: Path, estimator, args) -> tuple[int, int]:
    """Cache an arbitrary directory tree of videos.

    Subject is taken from the immediate parent directory name, so a layout of
    `<root>/<subject>/<clip>.avi` produces a valid subject-independent split.
    A flat directory would put every clip under one subject, which is why that
    case is rejected rather than silently accepted.
    """
    videos = sorted(
        p for pattern in ("*.mp4", "*.avi", "*.mov") for p in root.rglob(pattern)
    )
    print(f"found {len(videos)} videos under {root}")
    if videos and all(p.parent == root for p in videos):
        raise ValueError(
            "every video sits directly in the root, so subjects cannot be "
            "inferred. Arrange as <root>/<subject>/<clip>.mp4, or extend this "
            "script with a manifest."
        )

    done = failed = 0
    for index, path in enumerate(videos, 1):
        is_fall = "fall" in path.stem.lower() or "fall" in path.parent.name.lower()
        try:
            cache_video(
                video_path=path,
                out_dir=out,
                estimator=estimator,
                clip_id=f"{path.parent.name}_{path.stem}",
                subject=path.parent.name,
                label="fall" if is_fall else "adl",
                impact_frame=0 if is_fall else None,
                activity=path.stem,
                source=args.source or root.name,
                overwrite=args.overwrite,
                subject_policy=args.subject_policy,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            done += 1
        except Exception:
            failed += 1
            print(f"  [{index}/{len(videos)}] FAILED {path.name}")
            traceback.print_exc(limit=1)
    return done, failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="upfall", choices=["upfall", "generic"])
    parser.add_argument("--root", required=True, help="directory of source videos")
    parser.add_argument("--out", required=True, help="skeleton cache directory")
    parser.add_argument("--pose", default="rtmpose", help="rtmpose | yolo")
    parser.add_argument("--pose-device", default="cuda")
    parser.add_argument("--subject-policy", default="largest",
                        choices=["largest", "central", "most_confident"])
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth frame; the cached fps is divided to match")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source", default=None, help="dataset name recorded on each clip")
    args = parser.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    estimator = build_estimator(args.pose, device=args.pose_device)
    print(f"pose backend: {estimator.name} (frozen)")

    try:
        handler = cache_upfall if args.dataset == "upfall" else cache_generic
        done, failed = handler(root, out, estimator, args)
    finally:
        estimator.close()

    print(f"\ncached {done} clips ({failed} failed) -> {out}")
    stats = cache_stats(out)
    print(
        f"coverage: {stats['detected_joint_fraction']:.1%} of joints confidently "
        f"detected; {stats['frames_with_no_detection']} of {stats['total_frames']} "
        f"frames had no detection at all"
    )
    if stats["detected_joint_fraction"] < 0.75:
        print(
            "WARNING: low detection coverage. Check the subject policy and the "
            "confidence threshold before training - heavy interpolation will "
            "produce results that look fine and mean little."
        )
    if args.dataset == "upfall":
        print("\nnext: python scripts/annotate_onsets.py --cache", out)


if __name__ == "__main__":
    main()
