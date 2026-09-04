"""Stage C - slide a fixed-length window over a clip.

Training samples windows at a coarse stride for throughput; online inference
slides with stride 1 so a warning can fire on any frame. Both use the same
`Window` record, so the decision logic in `evaluation/` sees identical tensors in
either mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    """One `T`-frame window cut out of a clip."""

    clip_id: str
    start: int
    """Index of the window's first frame within the clip."""

    features: np.ndarray
    """`[C, T, V]` normalised features."""

    labels: np.ndarray
    """`[T]` per-frame labels, possibly containing `IGNORE_INDEX`."""

    tti: np.ndarray
    """`[T]` seconds until impact, `+inf` for ADL clips."""

    @property
    def end(self) -> int:
        """Index of the window's last frame (inclusive) within the clip."""
        return self.start + self.features.shape[1] - 1


def window_starts(num_frames: int, window: int, stride: int) -> list[int]:
    """Start indices of every full window, plus a final flush-right window.

    The trailing window keeps the last frames of a clip from being dropped,
    which matters because in a fall clip those are the frames around `t*`.
    """
    if window < 1 or stride < 1:
        raise ValueError(f"window and stride must be >= 1, got {window}, {stride}")
    if num_frames < window:
        return []
    starts = list(range(0, num_frames - window + 1, stride))
    last = num_frames - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def slice_windows(
    clip_id: str,
    features: np.ndarray,
    labels: np.ndarray,
    tti: np.ndarray,
    window: int,
    stride: int,
) -> list[Window]:
    """Cut a clip into windows.

    Args:
        clip_id: identifier carried through to the metrics, so a warning can be
            attributed back to the clip it fired on.
        features: `[C, T, V]`.
        labels: `[T]`.
        tti: `[T]` seconds to impact.
        window: window length in frames.
        stride: hop between consecutive windows (1 for online inference).

    Returns:
        A list of `Window`, empty if the clip is shorter than the window.
    """
    num_frames = features.shape[1]
    if labels.shape[0] != num_frames or tti.shape[0] != num_frames:
        raise ValueError(
            f"labels/tti length must match T={num_frames}, "
            f"got {labels.shape[0]}/{tti.shape[0]}"
        )
    return [
        Window(
            clip_id=clip_id,
            start=s,
            features=np.ascontiguousarray(features[:, s : s + window, :]),
            labels=np.ascontiguousarray(labels[s : s + window]),
            tti=np.ascontiguousarray(tti[s : s + window]),
        )
        for s in window_starts(num_frames, window, stride)
    ]


def pad_clip(
    features: np.ndarray,
    labels: np.ndarray,
    tti: np.ndarray,
    window: int,
    phases: np.ndarray | None = None,
):
    """Left-pad a clip that is shorter than one window by repeating frame 0.

    Used at inference so that a short clip still produces a score stream rather
    than nothing at all. Padded frames are labelled `IGNORE_INDEX` so they can
    never contribute to a metric.

    Every per-frame array must be padded together. Padding the features and
    labels but not the phases would leave the phase array shorter than the
    stream it annotates, and the misalignment would be silent - each phase would
    describe a frame `pad` positions later than the one it belongs to.
    """
    from .labels import IGNORE_INDEX

    num_frames = features.shape[1]
    if num_frames >= window:
        return (features, labels, tti) if phases is None else (features, labels, tti, phases)

    pad = window - num_frames
    features = np.concatenate([np.repeat(features[:, :1, :], pad, axis=1), features], axis=1)
    labels = np.concatenate([np.full(pad, IGNORE_INDEX, dtype=labels.dtype), labels])
    tti = np.concatenate([np.full(pad, tti[0], dtype=tti.dtype), tti])
    if phases is None:
        return features, labels, tti
    phases = np.concatenate([np.full(pad, IGNORE_INDEX, dtype=phases.dtype), phases])
    return features, labels, tti, phases
