"""Video -> cached skeletons. The one place pixels are read, and the last.

Everything after this module works from `.npz` skeleton files. A run that starts
from an existing cache never opens a video at all, which is both the privacy
guarantee (Section 19) and the reason training is cheap: a 10-second clip is
about 20 kB of skeleton rather than 30 MB of frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.clips import CACHE_SUFFIX, ClipRecord
from data.skeleton import NUM_JOINTS

from .base import PoseEstimator, empty_keypoints, select_subject, track_greedy


def _open_video(path: str | Path):
    """Open a video with OpenCV, with a useful error if it is not installed."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "reading video needs OpenCV (pip install opencv-python). "
            "Training and evaluation do not - they run from the skeleton cache."
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"could not open video: {path}")
    return capture


def extract_keypoints(
    video_path: str | Path,
    estimator: PoseEstimator,
    subject_policy: str = "largest",
    track: bool = True,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, float]:
    """Run the frozen estimator over a video.

    Args:
        video_path: the source video.
        estimator: a frozen `PoseEstimator`.
        subject_policy: how to choose the subject on the first frame.
        track: keep following the same person on later frames rather than
            re-applying the policy each frame.
        max_frames: stop early, for smoke tests.
        stride: keep every `stride`-th frame. Values above 1 change the
            effective frame rate, and the returned fps reflects that - lead time
            is reported in seconds, so this must not be silently ignored.

    Returns:
        `(keypoints [T, V, 3], fps)`. Frames with no detection are all-zero at
        zero confidence, which `data.normalize` interpolates across.
    """
    capture = _open_video(video_path)
    try:
        import cv2

        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frames: list[np.ndarray] = []
        previous = None
        index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride != 0:
                index += 1
                continue

            detections = estimator.detect(frame)
            chosen = (
                track_greedy(previous, detections)
                if track and previous is not None
                else select_subject(detections, frame.shape[:2], subject_policy)
            )
            frames.append(chosen.keypoints if chosen is not None else empty_keypoints())
            previous = chosen
            # The frame goes out of scope here and is never stored or written.

            index += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return np.stack(frames).astype(np.float32), source_fps / max(stride, 1)


def cache_video(
    video_path: str | Path,
    out_dir: str | Path,
    estimator: PoseEstimator,
    clip_id: str,
    subject: str,
    label: str = "adl",
    impact_frame: int | None = None,
    activity: str = "unknown",
    view: str = "default",
    source: str = "unknown",
    overwrite: bool = False,
    **extract_kwargs,
) -> Path:
    """Extract and cache one video as a `ClipRecord`.

    Skips work when the cache file already exists and `overwrite` is False, so
    an interrupted extraction over a large dataset can simply be re-run.
    """
    out_dir = Path(out_dir)
    target = out_dir / f"{clip_id}{CACHE_SUFFIX}"
    if target.exists() and not overwrite:
        return target

    keypoints, fps = extract_keypoints(video_path, estimator, **extract_kwargs)
    if keypoints.shape[1] != NUM_JOINTS:
        raise ValueError(
            f"{clip_id}: estimator returned {keypoints.shape[1]} joints, "
            f"expected {NUM_JOINTS}"
        )

    record = ClipRecord(
        clip_id=clip_id,
        subject=subject,
        keypoints=keypoints,
        fps=fps,
        label=label,
        impact_frame=impact_frame if label == "fall" else None,
        activity=activity,
        view=view,
        source=source,
    )
    return record.save(out_dir)


def cache_stats(cache_dir: str | Path) -> dict:
    """Coverage numbers for a skeleton cache: how much of it is real detections.

    A cache that looks complete but is 40% interpolated will train and evaluate
    without complaint and produce quietly wrong results, so this is worth
    looking at before the first training run.
    """
    from data.clips import load_cache
    from data.normalize import DEFAULT_CONF_THRESHOLD

    clips = load_cache(cache_dir)
    total = detected = 0
    empty_frames = 0
    for clip in clips:
        conf = clip.keypoints[..., 2]
        total += conf.size
        detected += int((conf >= DEFAULT_CONF_THRESHOLD).sum())
        empty_frames += int((conf.max(axis=1) <= 0.0).sum())
    return {
        "clips": len(clips),
        "detected_joint_fraction": detected / max(total, 1),
        "frames_with_no_detection": empty_frames,
        "total_frames": sum(c.num_frames for c in clips),
    }
