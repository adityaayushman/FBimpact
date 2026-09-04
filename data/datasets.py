"""Torch datasets over cached skeleton clips.

`WindowDataset` yields the fixed-length windows the model trains on.
`ClipDataset` yields whole clips, because evaluation is clip-level: lead time,
recall and the false-alarm rate are all defined over a clip's score stream, not
over shuffled windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import Augmenter
from .clips import ClipRecord
from .labels import IGNORE_INDEX, LabelConfig, frame_labels, time_to_impact
from .normalize import DEFAULT_CONF_THRESHOLD, centre_and_scale, interpolate_low_confidence
from .windows import Window, pad_clip, window_starts


@dataclass(frozen=True)
class FeatureConfig:
    """How a cached clip becomes model input."""

    window: int = 30
    stride: int = 5
    """Training stride. Inference always uses stride 1 (see `ClipDataset`)."""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    with_velocity: bool = True
    """False for the `- velocity features` ablation. Ignored when
    `feature_set` is set explicitly, which supersedes it."""

    feature_set: str | None = None
    """One of `data.features.FEATURE_SETS`. None keeps the original behaviour,
    selecting `xyv` or `xy` from `with_velocity`, so every existing config and
    checkpoint keeps meaning exactly what it did."""

    temporal_jitter: int = 0
    """Max random shift of the window start, in frames, at training time."""

    @property
    def resolved_feature_set(self) -> str:
        if self.feature_set is not None:
            return self.feature_set
        return "xyv" if self.with_velocity else "xy"

    @property
    def in_channels(self) -> int:
        from .features import channels_for

        return channels_for(self.resolved_feature_set)


def _prepare(
    clip: ClipRecord,
    features_cfg: FeatureConfig,
    label_cfg: LabelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gap-fill and centre one clip.

    Returns `(xy [T,V,2], labels [T], tti [T], phases [T])`. The phase array is
    all-`IGNORE_INDEX` when phases are switched off, so downstream code can pass
    it through unconditionally and the loss simply finds nothing to score.
    """
    xy, _ = interpolate_low_confidence(
        clip.keypoints[..., :2], clip.keypoints[..., 2], features_cfg.conf_threshold
    )
    xy, _, _ = centre_and_scale(xy)
    labels = frame_labels(clip.num_frames, clip.impact_frame, label_cfg)
    tti = time_to_impact(clip.num_frames, clip.impact_frame, clip.fps)

    if label_cfg.phases:
        from .phases import PhaseConfig, phase_labels

        phases = phase_labels(
            clip.num_frames, clip.impact_frame, clip.onset_frame,
            PhaseConfig(w_pre=label_cfg.w_pre,
                        include_grounded=label_cfg.post_impact == "ignore"),
        )
    else:
        phases = np.full(clip.num_frames, IGNORE_INDEX, dtype=np.int64)

    return xy, labels, tti, phases


def _to_features(xy: np.ndarray, fps: float, feature_set: str) -> np.ndarray:
    """`[T, V, 2]` -> `[C, T, V]` for the named feature set."""
    from .features import build_features

    return build_features(xy, fps=fps, feature_set=feature_set)


class WindowDataset(Dataset):
    """Windows for training.

    Augmentation is applied to the whole clip once per `__getitem__`, then the
    window is cut out of it. Doing it in that order keeps the velocity channel
    consistent with the augmented positions and lets temporal jitter shift the
    window without falling off the end of the clip.
    """

    def __init__(
        self,
        clips: list[ClipRecord],
        features_cfg: FeatureConfig | None = None,
        label_cfg: LabelConfig | None = None,
        augmenter: Augmenter | None = None,
        seed: int = 0,
    ) -> None:
        self.clips = list(clips)
        self.features_cfg = features_cfg or FeatureConfig()
        self.label_cfg = label_cfg or LabelConfig()
        self.augmenter = augmenter or Augmenter(enabled=False)
        self.seed = seed

        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.index: list[tuple[int, int]] = []
        for clip_idx, clip in enumerate(self.clips):
            starts = window_starts(
                clip.num_frames, self.features_cfg.window, self.features_cfg.stride
            )
            self.index.extend((clip_idx, s) for s in starts)
        if not self.index:
            raise ValueError(
                f"no windows of length {self.features_cfg.window} fit any of the "
                f"{len(self.clips)} clips - shorten the window or check the cache"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _clip_arrays(self, clip_idx: int):
        if clip_idx not in self._cache:
            self._cache[clip_idx] = _prepare(
                self.clips[clip_idx], self.features_cfg, self.label_cfg
            )
        return self._cache[clip_idx]

    def __getitem__(self, item: int) -> dict:
        clip_idx, start = self.index[item]
        clip = self.clips[clip_idx]
        xy, labels, tti, phases = self._clip_arrays(clip_idx)
        window = self.features_cfg.window

        # Seeded per item so a run is reproducible regardless of worker count.
        rng = np.random.default_rng((self.seed, item))

        jitter = self.features_cfg.temporal_jitter
        if jitter > 0:
            shift = int(rng.integers(-jitter, jitter + 1))
            start = int(np.clip(start + shift, 0, clip.num_frames - window))

        xy = self.augmenter(xy, rng)
        features = _to_features(xy, clip.fps, self.features_cfg.resolved_feature_set)

        return {
            "features": torch.from_numpy(
                np.ascontiguousarray(features[:, start : start + window, :])
            ),
            "labels": torch.from_numpy(
                np.ascontiguousarray(labels[start : start + window])
            ),
            "tti": torch.from_numpy(np.ascontiguousarray(tti[start : start + window])),
            "phases": torch.from_numpy(
                np.ascontiguousarray(phases[start : start + window])
            ),
            "clip_idx": clip_idx,
            "start": start,
        }

    def positive_fraction(self) -> float:
        """Fraction of non-ignored frames that are positive, for class weighting."""
        pos = total = 0
        for clip_idx, start in self.index:
            _, labels, _, _ = self._clip_arrays(clip_idx)
            chunk = labels[start : start + self.features_cfg.window]
            counted = chunk[chunk != IGNORE_INDEX]
            pos += int((counted == 1).sum())
            total += int(counted.size)
        return pos / max(total, 1)


class ClipDataset(Dataset):
    """Whole clips for evaluation and inference.

    Each item is one clip's full window stack at stride 1, so the model produces
    a score for every frame and the decision logic sees the same stream it would
    see online.
    """

    def __init__(
        self,
        clips: list[ClipRecord],
        features_cfg: FeatureConfig | None = None,
        label_cfg: LabelConfig | None = None,
    ) -> None:
        self.clips = list(clips)
        self.features_cfg = features_cfg or FeatureConfig()
        self.label_cfg = label_cfg or LabelConfig()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, item: int) -> dict:
        clip = self.clips[item]
        xy, labels, tti, phases = _prepare(clip, self.features_cfg, self.label_cfg)
        features = _to_features(xy, clip.fps, self.features_cfg.resolved_feature_set)
        features, labels, tti, phases = pad_clip(
            features, labels, tti, self.features_cfg.window, phases
        )

        window = self.features_cfg.window
        starts = window_starts(features.shape[1], window, stride=1)
        stack = np.stack([features[:, s : s + window, :] for s in starts])

        return {
            "clip": clip,
            "windows": torch.from_numpy(np.ascontiguousarray(stack)),  # [N, C, T, V]
            "starts": np.asarray(starts, dtype=np.int64),
            "labels": labels,
            "tti": tti,
            "phases": phases,
            "n_padded": features.shape[1] - clip.num_frames,
        }


def collate_windows(batch: list[dict]) -> dict:
    """Stack window items into `[N, C, T, V, M]` with `M = 1` person."""
    return {
        "features": torch.stack([b["features"] for b in batch]).unsqueeze(-1),
        "labels": torch.stack([b["labels"] for b in batch]),
        "tti": torch.stack([b["tti"] for b in batch]),
        "phases": torch.stack([b["phases"] for b in batch]),
        "clip_idx": torch.tensor([b["clip_idx"] for b in batch], dtype=torch.long),
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
    }


def collate_clips(batch: list[dict]) -> dict:
    """Clips are evaluated one at a time; batching them buys nothing."""
    if len(batch) != 1:
        raise ValueError("ClipDataset must be loaded with batch_size=1")
    return batch[0]


__all__ = [
    "FeatureConfig",
    "WindowDataset",
    "ClipDataset",
    "collate_windows",
    "collate_clips",
    "Window",
]
