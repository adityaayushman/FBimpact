"""Stages A-C: cached skeletons, normalisation, pre-impact labelling, windowing."""

from .augment import Augmenter
from .clips import ClipRecord, load_cache, summarise
from .datasets import ClipDataset, FeatureConfig, WindowDataset, collate_clips, collate_windows
from .labels import IGNORE_INDEX, LabelConfig, frame_labels, time_to_impact
from .normalize import normalise_clip
from .skeleton import BONES, JOINT_NAMES, NUM_JOINTS, adjacency_matrix
from .splits import Split, assert_subject_disjoint, filter_clips, single_split, subject_folds
from .windows import Window, slice_windows, window_starts

__all__ = [
    "Augmenter",
    "BONES",
    "ClipDataset",
    "ClipRecord",
    "FeatureConfig",
    "IGNORE_INDEX",
    "JOINT_NAMES",
    "LabelConfig",
    "NUM_JOINTS",
    "Split",
    "Window",
    "WindowDataset",
    "adjacency_matrix",
    "assert_subject_disjoint",
    "collate_clips",
    "collate_windows",
    "filter_clips",
    "frame_labels",
    "load_cache",
    "normalise_clip",
    "single_split",
    "slice_windows",
    "subject_folds",
    "summarise",
    "time_to_impact",
    "window_starts",
]
