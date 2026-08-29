"""Config -> datasets. The single place a split is constructed.

Centralising this matters more than it looks: `train.py`, `eval.py` and the
ablation runner all have to build *the same* split from the same config, or an
ablation ends up being compared against a baseline that saw different subjects.
Every path in here ends at `assert_subject_disjoint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .augment import Augmenter
from .clips import ClipRecord, load_cache, summarise
from .datasets import ClipDataset, FeatureConfig, WindowDataset
from .labels import LabelConfig
from .splits import Split, assert_subject_disjoint, filter_clips, single_split, subject_folds


@dataclass
class DataBundle:
    """Everything a run needs, already split and configured."""

    train: WindowDataset
    val_clips: ClipDataset
    test_clips: ClipDataset
    split: Split
    features_cfg: FeatureConfig
    label_cfg: LabelConfig
    stats: dict

    @property
    def fps(self) -> float:
        clips = self.test_clips.clips or self.val_clips.clips
        return float(clips[0].fps) if clips else 30.0


def load_clips(cfg: dict) -> list[ClipRecord]:
    """Load the skeleton cache named by the config, applying onset annotations."""
    data_cfg = cfg.get("data", {})
    cache_dir = Path(data_cfg.get("cache_dir", "data/cache/synthetic"))
    clips = load_cache(cache_dir)

    onsets_path = data_cfg.get("onsets")
    if onsets_path:
        from .upfall import attach_labels, read_onsets

        onsets = read_onsets(onsets_path)
        if not onsets:
            raise FileNotFoundError(
                f"data.onsets points at {onsets_path}, which has no usable rows. "
                f"Run scripts/annotate_onsets.py before training on real fall data."
            )
        clips, dropped = attach_labels(clips, onsets)
        if dropped:
            print(
                f"[data] dropped {len(dropped)} fall clips with no annotated impact "
                f"frame (first few: {dropped[:5]})"
            )
    else:
        missing = [c.clip_id for c in clips if c.is_fall and c.impact_frame is None]
        if missing:
            raise ValueError(
                f"{len(missing)} fall clips have no impact frame and no "
                f"data.onsets file was configured; lead time cannot be measured"
            )

    if not clips:
        raise ValueError(f"no usable clips left in {cache_dir}")
    return clips


def build_split(cfg: dict, clips: list[ClipRecord]) -> Split:
    """Construct the subject-independent split named by the config."""
    split_cfg = cfg.get("data", {}).get("split", {})
    mode = split_cfg.get("mode", "single")
    seed = int(split_cfg.get("seed", 0))

    if mode == "single":
        split = single_split(
            clips,
            test_fraction=float(split_cfg.get("test_fraction", 0.2)),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
            seed=seed,
        )
    elif mode == "folds":
        folds = subject_folds(
            clips,
            num_folds=int(split_cfg.get("num_folds", 5)),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
            seed=seed,
        )
        index = int(split_cfg.get("fold", 0))
        if not 0 <= index < len(folds):
            raise ValueError(f"fold {index} out of range for {len(folds)} folds")
        split = folds[index]
    else:
        raise ValueError(f"data.split.mode must be 'single' or 'folds', got {mode!r}")

    assert_subject_disjoint(split)
    return split


def build_datasets(cfg: dict, clips: list[ClipRecord] | None = None) -> DataBundle:
    """Build train/val/test datasets from a resolved config."""
    clips = clips if clips is not None else load_clips(cfg)
    split = build_split(cfg, clips)

    features_cfg = FeatureConfig(**dict(cfg.get("features", {})))
    label_cfg = LabelConfig(**dict(cfg.get("labels", {})))

    train_clips = filter_clips(clips, split.train)
    val_clips = filter_clips(clips, split.val)
    test_clips = filter_clips(clips, split.test)
    for name, subset in (("train", train_clips), ("val", val_clips), ("test", test_clips)):
        if not subset:
            raise ValueError(f"the {name} split contains no clips - check data.split")

    seed = int(cfg.get("run", {}).get("seed", 0))
    return DataBundle(
        train=WindowDataset(
            train_clips,
            features_cfg=features_cfg,
            label_cfg=label_cfg,
            augmenter=Augmenter.from_config(cfg.get("augment"), enabled=True),
            seed=seed,
        ),
        val_clips=ClipDataset(val_clips, features_cfg, label_cfg),
        test_clips=ClipDataset(test_clips, features_cfg, label_cfg),
        split=split,
        features_cfg=features_cfg,
        label_cfg=label_cfg,
        stats={
            "all": summarise(clips),
            "train": summarise(train_clips),
            "val": summarise(val_clips),
            "test": summarise(test_clips),
        },
    )
