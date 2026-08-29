"""Stage A - the frozen perception interface.

Pose estimation is explicitly *not* a contribution (Section 5), so this package
is a thin, swappable wrapper and nothing more. What it does own is the **privacy
boundary**: a `PoseEstimator` takes frames and returns keypoints, and nothing in
this package is allowed to persist a frame, a crop or a path that could be
decoded back into one. Section 19's "only skeletons are stored, not video" is a
property of the code, not a promise in the paper.

It also owns subject selection. Public fall datasets are single-subject, but a
detector will still occasionally return a reflection, a shadow or a piece of
furniture, so a policy has to be applied and stated rather than left to whichever
detection happens to come first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from data.skeleton import NUM_JOINTS

SUBJECT_POLICIES = ("largest", "central", "most_confident")


@dataclass(frozen=True)
class Detection:
    """One detected person in one frame."""

    keypoints: np.ndarray
    """`[V, 3]` of `(x, y, confidence)` in pixels."""

    box: tuple[float, float, float, float] | None = None
    """`(x1, y1, x2, y2)`, when the backend provides it."""

    score: float = 1.0
    """Detector confidence for the person, not for individual joints."""

    @property
    def area(self) -> float:
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        visible = self.keypoints[self.keypoints[:, 2] > 0.3, :2]
        if visible.shape[0] < 2:
            return 0.0
        span = visible.max(axis=0) - visible.min(axis=0)
        return float(span[0] * span[1])

    def centre(self) -> np.ndarray:
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        visible = self.keypoints[self.keypoints[:, 2] > 0.3, :2]
        if visible.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)
        return visible.mean(axis=0).astype(np.float32)


class PoseEstimator(ABC):
    """A frozen 2D pose estimator.

    Implementations must not fine-tune, and must not retain frames.
    """

    name: str = "abstract"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Estimate poses in one BGR/RGB frame `[H, W, 3]`."""

    def close(self) -> None:
        """Release backend resources. Safe to call more than once."""

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def select_subject(
    detections: list[Detection],
    frame_shape: tuple[int, int],
    policy: str = "largest",
) -> Detection | None:
    """Pick the one person to track (Section 8, Stage A).

    Args:
        detections: everything the backend found in this frame.
        frame_shape: `(height, width)`, needed by the `central` policy.
        policy: `"largest"` (nearest the camera, the default and the one used
            for UP-Fall), `"central"` (nearest the frame centre) or
            `"most_confident"`.

    Returns:
        The chosen detection, or None if nothing was found.
    """
    if policy not in SUBJECT_POLICIES:
        raise ValueError(f"policy must be one of {SUBJECT_POLICIES}, got {policy!r}")
    if not detections:
        return None

    if policy == "largest":
        return max(detections, key=lambda d: d.area)
    if policy == "most_confident":
        return max(detections, key=lambda d: d.score)

    height, width = frame_shape
    centre = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    return min(detections, key=lambda d: float(np.linalg.norm(d.centre() - centre)))


def empty_keypoints() -> np.ndarray:
    """A fully-missing frame: zero coordinates at zero confidence.

    Emitted when the detector finds nobody. The zero confidence is what matters -
    `data.normalize` will interpolate across it rather than treat the origin as
    an observed pose.
    """
    return np.zeros((NUM_JOINTS, 3), dtype=np.float32)


def track_greedy(
    previous: Detection | None, detections: list[Detection], max_jump: float = 0.35
) -> Detection | None:
    """Keep following the same person across frames.

    Chooses the detection closest to the previous one, rejecting jumps larger
    than `max_jump` of the frame diagonal - which is how a tracker ends up
    switching from the faller to a bystander halfway through a clip, producing a
    skeleton sequence that no labelling scheme can describe.
    """
    if not detections:
        return None
    if previous is None:
        return max(detections, key=lambda d: d.area)

    anchor = previous.centre()
    scale = max(float(np.hypot(*anchor)), 1.0)
    best = min(detections, key=lambda d: float(np.linalg.norm(d.centre() - anchor)))
    if float(np.linalg.norm(best.centre() - anchor)) > max_jump * scale:
        return max(detections, key=lambda d: d.area)
    return best
