"""Stages G-H: decision logic, lead time, and the anticipation metrics."""

from .decision import DecisionConfig, OnlineTrigger, all_triggers, first_trigger, sweep_grid
from .metrics import (
    AnticipationReport,
    ClipOutcome,
    ClipScores,
    evaluate,
    evaluate_clip,
    operating_curve,
    select_operating_point,
)
from .runner import score_clip, score_dataset

__all__ = [
    "AnticipationReport",
    "ClipOutcome",
    "ClipScores",
    "DecisionConfig",
    "OnlineTrigger",
    "all_triggers",
    "evaluate",
    "evaluate_clip",
    "first_trigger",
    "operating_curve",
    "score_clip",
    "score_dataset",
    "select_operating_point",
    "sweep_grid",
]
