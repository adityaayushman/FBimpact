"""Stage B - normalise raw skeletons and add motion features.

Input is whatever the frozen pose estimator produced: `[T, V, 3]` of
`(x, y, confidence)` in pixel coordinates. Output is `[C, T, V]` with
`C = (x, y, vx, vy)` in a person-centred, scale-free frame, which is what the
temporal model consumes.

Three things happen here, and each one exists to remove a nuisance variable that
would otherwise let the model cheat:

* **Translation** to the mid-hip origin removes where in the room the person is.
* **Scaling** by torso length removes how far they are from the camera.
* **Confidence gating + interpolation** removes the pose estimator's dropouts,
  which are frequent exactly when a person is on the floor.
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    CENTRE_JOINTS,
    NUM_JOINTS,
    TORSO_BOTTOM_JOINTS,
    TORSO_TOP_JOINTS,
)

# Below this keypoint confidence a joint is treated as missing rather than
# trusted; pose estimators emit confident-looking garbage far less often than
# they emit low-confidence garbage.
DEFAULT_CONF_THRESHOLD = 0.30

# Guards against a degenerate torso length (person seen end-on, or a badly
# collapsed detection) blowing the normalised coordinates up to huge values.
MIN_TORSO_PIXELS = 1e-3


def interpolate_low_confidence(
    xy: np.ndarray,
    conf: np.ndarray,
    threshold: float = DEFAULT_CONF_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate joints whose confidence falls below `threshold`.

    Args:
        xy: `[T, V, 2]` pixel coordinates.
        conf: `[T, V]` keypoint confidences.
        threshold: confidence below which a joint is considered missing.

    Returns:
        `(xy_filled, valid)` where `valid [T, V]` marks frames that were
        observed rather than reconstructed. Joints missing for an entire clip
        stay at zero and are reported invalid, so downstream code can decide
        whether to drop the clip.
    """
    xy = np.asarray(xy, dtype=np.float32).copy()
    valid = np.asarray(conf, dtype=np.float32) >= threshold
    n_frames = xy.shape[0]
    frames = np.arange(n_frames, dtype=np.float32)

    for v in range(xy.shape[1]):
        observed = valid[:, v]
        n_observed = int(observed.sum())
        if n_observed == n_frames:
            continue
        if n_observed == 0:
            xy[:, v, :] = 0.0
            continue
        for c in range(2):
            # np.interp holds the endpoint value outside the observed range,
            # which is the behaviour we want at clip boundaries.
            xy[:, v, c] = np.interp(frames, frames[observed], xy[observed, v, c])
    return xy, valid


def centre_and_scale(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate to the mid-hip origin and divide by torso length.

    Args:
        xy: `[T, V, 2]` pixel coordinates, already gap-filled.

    Returns:
        `(normalised [T, V, 2], centre [T, 2], scale [T])`. The centre and scale
        are returned so an explanation or a visualisation can be mapped back to
        image space.
    """
    xy = np.asarray(xy, dtype=np.float32)
    centre = xy[:, CENTRE_JOINTS, :].mean(axis=1)                       # [T, 2]
    top = xy[:, TORSO_TOP_JOINTS, :].mean(axis=1)                       # [T, 2]
    bottom = xy[:, TORSO_BOTTOM_JOINTS, :].mean(axis=1)                 # [T, 2]
    scale = np.linalg.norm(top - bottom, axis=-1)                       # [T]

    # A per-frame scale would cancel out exactly the vertical collapse we are
    # trying to detect, so use one robust scale for the whole clip.
    clip_scale = float(np.median(scale[scale > MIN_TORSO_PIXELS])) if np.any(
        scale > MIN_TORSO_PIXELS
    ) else 1.0
    clip_scale = max(clip_scale, MIN_TORSO_PIXELS)

    normalised = (xy - centre[:, None, :]) / clip_scale
    return normalised.astype(np.float32), centre.astype(np.float32), scale.astype(np.float32)


def add_velocity(xy: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Append per-joint velocity, giving `[C=4, T, V]`.

    The velocity is a causal backward difference (`x_t - x_{t-1}`), never a
    centred difference: a centred difference would leak one frame of the future
    into every feature, which for an anticipation task means leaking the fall
    itself. Scaled to units per second so the feature is frame-rate independent.
    """
    xy = np.asarray(xy, dtype=np.float32)
    velocity = np.zeros_like(xy)
    velocity[1:] = (xy[1:] - xy[:-1]) * float(fps)
    velocity[0] = velocity[1] if len(xy) > 1 else 0.0

    features = np.concatenate([xy, velocity], axis=-1)  # [T, V, 4]
    return np.transpose(features, (2, 0, 1)).astype(np.float32)  # [C, T, V]


def normalise_clip(
    keypoints: np.ndarray,
    fps: float = 30.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    with_velocity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the whole of Stage B on one clip.

    Args:
        keypoints: `[T, V, 3]` of `(x, y, confidence)` in pixels.
        fps: frame rate, used to express velocity per second.
        conf_threshold: confidence below which a joint is interpolated.
        with_velocity: set False for the `- velocity features` ablation, which
            yields `C = 2` instead of `C = 4`.

    Returns:
        `(features [C, T, V], valid [T, V])`.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_JOINTS or keypoints.shape[2] != 3:
        raise ValueError(
            f"expected keypoints of shape [T, {NUM_JOINTS}, 3], got {keypoints.shape}"
        )

    xy, valid = interpolate_low_confidence(
        keypoints[..., :2], keypoints[..., 2], conf_threshold
    )
    xy, _, _ = centre_and_scale(xy)

    if with_velocity:
        features = add_velocity(xy, fps=fps)
    else:
        features = np.transpose(xy, (2, 0, 1)).astype(np.float32)  # [2, T, V]
    return features, valid
