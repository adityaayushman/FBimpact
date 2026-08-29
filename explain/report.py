"""Glue: from a fired warning to a grounded, tested explanation.

Given a clip's score stream and the frame the warning fired on, this locates the
exact window a live system would have been holding at that moment, ranks its
joints (Stage E) and runs the deletion/insertion test on it (Stage F). Doing the
lookup here, once, keeps every consumer - `eval.py`, `infer.py`, the ablations -
explaining the same tensor the decision was made from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data.clips import ClipRecord
from data.datasets import ClipDataset
from evaluation.decision import DecisionConfig
from evaluation.metrics import ClipOutcome, ClipScores, evaluate_clip

from .faithfulness import FaithfulnessCurves, FaithfulnessReport, aggregate, faithfulness_curves
from .relevance import JointRelevance, joint_relevance


@dataclass
class GroundedWarning:
    """One warning together with the evidence that justifies it."""

    clip_id: str
    trigger_frame: int
    lead_time: float | None
    score: float
    relevance: JointRelevance
    curves: FaithfulnessCurves | None = None

    def message(self, k: int = 3) -> str:
        """The line a caregiver-facing alert would carry."""
        lead = f"{self.lead_time:.2f}s before impact" if self.lead_time is not None else "now"
        joints = ", ".join(f"{n}" for n, _ in self.relevance.top_k(k))
        return (
            f"[{self.clip_id}] fall imminent ({lead}, p={self.score:.2f}) - "
            f"evidence: {self.relevance.phrase(k)} [{joints}]"
        )


def locate_window(item: dict, clip_frame: int) -> tuple[int, int]:
    """Window index and in-window position for a clip frame.

    Returns the window that **ends** on `clip_frame`, matching the score the
    decision rule acted on. Early frames of a clip fall inside window 0, so the
    in-window position is returned rather than assumed to be the last.
    """
    window_len = int(item["windows"].shape[2])
    padded_frame = int(clip_frame) + int(item["n_padded"])
    window_index = padded_frame - (window_len - 1)
    if window_index < 0:
        return 0, padded_frame
    max_index = int(item["windows"].shape[0]) - 1
    window_index = min(window_index, max_index)
    return window_index, padded_frame - window_index


def explain_warning(
    model: torch.nn.Module,
    item: dict,
    clip_frame: int,
    device: torch.device,
    method: str = "attention",
    score: float | None = None,
    lead_time: float | None = None,
    with_faithfulness: bool = True,
    baseline: str = "zero",
    num_random: int = 5,
    seed: int = 0,
    mean_pose: torch.Tensor | None = None,
) -> GroundedWarning:
    """Explain the warning that fired on `clip_frame` of this clip."""
    window_index, position = locate_window(item, clip_frame)
    window = item["windows"][window_index]                 # [C, T, V]
    clip: ClipRecord = item["clip"]

    relevance = joint_relevance(model, window, position, device, method=method)

    curves = None
    if with_faithfulness:
        curves = faithfulness_curves(
            model=model,
            window=window,
            frame=position,
            relevance=relevance.scores,
            device=device,
            baseline=baseline,
            num_random=num_random,
            seed=seed,
            mean_pose=mean_pose,
            method=method,
        )

    if score is None:
        with torch.no_grad():
            from .relevance import _prepare_window

            score = float(
                torch.sigmoid(model(_prepare_window(window, device)))[0, position].item()
            )

    return GroundedWarning(
        clip_id=clip.clip_id,
        trigger_frame=int(clip_frame),
        lead_time=lead_time,
        score=float(score),
        relevance=relevance,
        curves=curves,
    )


def explain_dataset(
    model: torch.nn.Module,
    dataset: ClipDataset,
    scored: list[ClipScores],
    decision: DecisionConfig,
    w_pre: int,
    device: torch.device,
    method: str = "attention",
    max_warnings: int | None = 100,
    falls_only: bool = True,
    with_faithfulness: bool = True,
    baseline: str = "zero",
    num_random: int = 5,
    seed: int = 0,
) -> tuple[list[GroundedWarning], FaithfulnessReport | None]:
    """Explain every warning in a scored dataset and aggregate faithfulness.

    Args:
        model: the trained model.
        dataset: the same `ClipDataset` the scores came from, in the same order.
        scored: per-clip score streams from `evaluation.runner.score_dataset`.
        decision: the operating point chosen on validation.
        w_pre: imminent-window length in frames.
        device: where to run.
        method: relevance method; see `explain.relevance.METHODS`.
        max_warnings: cap on explanations, since the deletion/insertion test
            costs `(1 + num_random) x 2 x (V + 1)` forward passes each. None
            explains everything.
        falls_only: explain correct warnings on fall clips only. Setting False
            also explains false alarms, which is the more interesting error
            analysis but is not what the headline faithfulness number is over.
        with_faithfulness: run Stage F as well as Stage E.
        baseline: joint-removal baseline for the faithfulness test.
        num_random: random control orderings per warning.
        seed: seed for the random controls.

    Returns:
        `(warnings, faithfulness_report)`; the report is None when no warning
        fired or `with_faithfulness` is False.
    """
    if len(dataset) != len(scored):
        raise ValueError(
            f"dataset has {len(dataset)} clips but {len(scored)} score streams were given"
        )

    was_training = model.training
    model.eval().to(device)

    # `baseline="mean"` needs a dataset mean pose; compute it once here rather
    # than letting apply_baseline raise deep inside the per-warning loop.
    mean_pose = dataset_mean_pose(dataset) if baseline == "mean" else None

    warnings: list[GroundedWarning] = []
    for index, item_scores in enumerate(scored):
        if max_warnings is not None and len(warnings) >= max_warnings:
            break
        outcome: ClipOutcome = evaluate_clip(item_scores, decision, w_pre)
        if outcome.trigger_frame is None:
            continue
        if falls_only and not (outcome.is_fall and outcome.warned):
            continue

        item = dataset[index]
        warnings.append(
            explain_warning(
                model=model,
                item=item,
                clip_frame=outcome.trigger_frame,
                device=device,
                method=method,
                score=float(item_scores.scores[outcome.trigger_frame]),
                lead_time=outcome.lead_time,
                with_faithfulness=with_faithfulness,
                baseline=baseline,
                num_random=num_random,
                seed=seed + index,
                mean_pose=mean_pose,
            )
        )

    if was_training:
        model.train()

    curves = [w.curves for w in warnings if w.curves is not None]
    report = aggregate(curves, baseline=baseline) if curves else None
    return warnings, report


def dataset_mean_pose(dataset: ClipDataset, max_clips: int = 50) -> torch.Tensor:
    """Mean feature vector per joint, `[C, V]`, for `baseline="mean"`."""
    totals = None
    count = 0
    for index in range(min(len(dataset), max_clips)):
        windows = dataset[index]["windows"]                  # [N, C, T, V]
        summed = windows.sum(dim=(0, 2))                     # [C, V]
        totals = summed if totals is None else totals + summed
        count += windows.shape[0] * windows.shape[2]
    if totals is None:
        raise ValueError("empty dataset - cannot compute a mean pose")
    return (totals / max(count, 1)).float()
