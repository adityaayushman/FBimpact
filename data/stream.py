"""Online feature extraction, for real-time inference.

`data.normalize` is the offline path: it interpolates a missing joint from the
frames on *both* sides of the gap and divides by the median torso length of the
whole clip. Neither is available live, so this module does the online
equivalents - hold the last observed value, and track the torso length with a
running median over recent frames.

The two paths therefore do not produce bit-identical features, and that gap is
worth being explicit about: an offline number is a mild upper bound on live
behaviour, because offline gap-filling has information the live system does not.
`compare_offline` exists to measure the difference on real clips rather than
leave it as a caveat, so the paper can state it as a quantity.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .normalize import DEFAULT_CONF_THRESHOLD, MIN_TORSO_PIXELS
from .skeleton import (
    CENTRE_JOINTS,
    NUM_JOINTS,
    TORSO_BOTTOM_JOINTS,
    TORSO_TOP_JOINTS,
)


class SkeletonStream:
    """Frame-by-frame normalisation and windowing for live inference.

    Holds `window` frames of features and nothing else - a few kilobytes,
    independent of how long the system has been running.
    """

    def __init__(
        self,
        window: int = 30,
        fps: float = 30.0,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        with_velocity: bool = True,
        scale_history: int = 90,
    ) -> None:
        """
        Args:
            window: `T`, must match the trained model's window.
            fps: frame rate, for the velocity scaling.
            conf_threshold: confidence below which a joint is held rather than trusted.
            with_velocity: must match the trained model's channel count.
            scale_history: frames of torso-length history for the running median.
                Long enough to be stable, short enough to follow a person walking
                towards or away from the camera.
        """
        self.window = window
        self.fps = float(fps)
        self.conf_threshold = conf_threshold
        self.with_velocity = with_velocity
        self.channels = 4 if with_velocity else 2
        self._scales: deque[float] = deque(maxlen=scale_history)
        self.reset()

    def reset(self) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self._last_xy: np.ndarray | None = None
        self._last_norm: np.ndarray | None = None
        self._scales.clear()
        self.frame_index = -1

    @property
    def ready(self) -> bool:
        """True once a full window has been seen."""
        return len(self._buffer) == self.window

    def push(self, keypoints: np.ndarray) -> np.ndarray | None:
        """Add one frame's `[V, 3]` keypoints; return the current `[C, T, V]` window.

        Returns None until a full window has accumulated. Before that a live
        system has no basis for a decision, and padding the buffer to force an
        early score would mean warning on evidence that does not exist.
        """
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.shape != (NUM_JOINTS, 3):
            raise ValueError(f"expected [{NUM_JOINTS}, 3] keypoints, got {keypoints.shape}")
        self.frame_index += 1

        xy = keypoints[:, :2].copy()
        confident = keypoints[:, 2] >= self.conf_threshold

        if self._last_xy is None:
            # First frame: nothing to hold, so an unconfident joint stays where
            # the detector put it and is corrected on the next confident frame.
            self._last_xy = xy.copy()
        else:
            xy[~confident] = self._last_xy[~confident]
            self._last_xy = xy.copy()

        centre = xy[list(CENTRE_JOINTS)].mean(axis=0)
        top = xy[list(TORSO_TOP_JOINTS)].mean(axis=0)
        bottom = xy[list(TORSO_BOTTOM_JOINTS)].mean(axis=0)
        torso = float(np.linalg.norm(top - bottom))
        if torso > MIN_TORSO_PIXELS:
            self._scales.append(torso)
        scale = float(np.median(self._scales)) if self._scales else 1.0
        scale = max(scale, MIN_TORSO_PIXELS)

        normalised = ((xy - centre) / scale).astype(np.float32)      # [V, 2]

        if self.with_velocity:
            previous = self._last_norm if self._last_norm is not None else normalised
            velocity = (normalised - previous) * self.fps
            features = np.concatenate([normalised, velocity], axis=1)  # [V, 4]
        else:
            features = normalised
        self._last_norm = normalised

        self._buffer.append(features.T.astype(np.float32))            # [C, V]
        if not self.ready:
            return None
        return np.stack(self._buffer, axis=1)                         # [C, T, V]


def compare_offline(clip, window: int = 30, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> dict:
    """Quantify how far the online features drift from the offline ones.

    Runs both paths over the same clip and reports the mean and maximum absolute
    difference per channel. A small number here is what licenses reporting
    offline metrics as an estimate of live performance; a large one is a finding
    that belongs in the paper.
    """
    from .normalize import normalise_clip

    offline, _ = normalise_clip(clip.keypoints, fps=clip.fps, conf_threshold=conf_threshold)
    stream = SkeletonStream(window=window, fps=clip.fps, conf_threshold=conf_threshold)

    online_frames: list[np.ndarray] = []
    for t in range(clip.num_frames):
        result = stream.push(clip.keypoints[t])
        if result is not None:
            online_frames.append(result[:, -1, :])                    # [C, V] at frame t

    if not online_frames:
        return {"frames_compared": 0}

    start = clip.num_frames - len(online_frames)
    online = np.stack(online_frames, axis=1)                          # [C, T', V]
    reference = offline[:, start:, :]
    delta = np.abs(online - reference)

    return {
        "frames_compared": len(online_frames),
        "mean_abs_diff": float(delta.mean()),
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff_position": float(delta[:2].mean()),
        "mean_abs_diff_velocity": float(delta[2:].mean()) if delta.shape[0] > 2 else 0.0,
    }
