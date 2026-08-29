"""Stages E-F: joint relevance and the deletion/insertion faithfulness test."""

from .faithfulness import (
    BASELINES,
    FaithfulnessCurves,
    FaithfulnessReport,
    aggregate,
    apply_baseline,
    faithfulness_curves,
)
from .relevance import METHODS, JointRelevance, joint_relevance
from .report import GroundedWarning, dataset_mean_pose, explain_dataset, explain_warning

__all__ = [
    "BASELINES",
    "FaithfulnessCurves",
    "FaithfulnessReport",
    "GroundedWarning",
    "JointRelevance",
    "METHODS",
    "aggregate",
    "apply_baseline",
    "dataset_mean_pose",
    "explain_dataset",
    "explain_warning",
    "faithfulness_curves",
    "joint_relevance",
]
