"""Section 9 - pre-impact labelling.

For a fall clip with impact frame `t*`, the imminent window is `[t* - W_pre, t*]`
and its frames are positive; everything before it is negative; every frame of an
ADL clip is negative.

One case the proposal leaves open is what happens **after** `t*`. Those frames
are neither imminent nor normal - the person is already on the floor - and
labelling them positive would quietly turn the anticipation task back into
post-fall detection, which is exactly what the project is trying not to do. The
default here is therefore to **ignore** them: they are masked out of the loss and
out of the frame-level metrics. `POST_IMPACT_POLICIES` keeps the two alternatives
available so the choice can be reported as a sensitivity check rather than
buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE_INDEX = -1

POST_IMPACT_POLICIES = ("ignore", "negative", "positive")


@dataclass(frozen=True)
class LabelConfig:
    """Parameters of the pre-impact labelling scheme."""

    w_pre: int = 20
    """Length of the imminent window in frames (~0.67 s at 30 fps)."""

    post_impact: str = "ignore"
    """One of `POST_IMPACT_POLICIES`; see the module docstring."""

    def __post_init__(self) -> None:
        if self.w_pre < 1:
            raise ValueError(f"w_pre must be >= 1, got {self.w_pre}")
        if self.post_impact not in POST_IMPACT_POLICIES:
            raise ValueError(
                f"post_impact must be one of {POST_IMPACT_POLICIES}, got {self.post_impact!r}"
            )


def frame_labels(
    num_frames: int,
    impact_frame: int | None,
    config: LabelConfig | None = None,
) -> np.ndarray:
    """Per-frame labels for one clip.

    Args:
        num_frames: clip length `T`.
        impact_frame: `t*`, the first frame of ground contact, or None for an
            ADL clip (every frame negative).
        config: labelling parameters.

    Returns:
        `[T]` int array of 0 (normal), 1 (fall imminent) or `IGNORE_INDEX`.
    """
    config = config or LabelConfig()
    labels = np.zeros(num_frames, dtype=np.int64)
    if impact_frame is None:
        return labels

    if not 0 <= impact_frame < num_frames:
        raise ValueError(
            f"impact_frame {impact_frame} outside clip of {num_frames} frames"
        )

    start = max(0, impact_frame - config.w_pre)
    labels[start : impact_frame + 1] = 1

    if config.post_impact == "ignore":
        labels[impact_frame + 1 :] = IGNORE_INDEX
    elif config.post_impact == "positive":
        labels[impact_frame + 1 :] = 1
    # "negative" needs no action - they are already zero.

    return labels


def time_to_impact(
    num_frames: int,
    impact_frame: int | None,
    fps: float = 30.0,
) -> np.ndarray:
    """Seconds until impact for every frame, `+inf` for an ADL clip.

    Positive before impact, negative after. Consumed by the time-weighted loss,
    which uses it to weight earlier correct warnings more heavily.
    """
    if impact_frame is None:
        return np.full(num_frames, np.inf, dtype=np.float32)
    frames = np.arange(num_frames, dtype=np.float32)
    return ((float(impact_frame) - frames) / float(fps)).astype(np.float32)
