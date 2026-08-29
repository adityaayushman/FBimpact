"""Section 15 - anticipation metrics.

Recall leads, because a missed fall is the costly error. Everything here is
computed clip-by-clip from a per-frame score stream, never from shuffled
windows, because lead time and the false-alarm rate only exist at clip level.

One judgement the proposal leaves open is what to do with a warning that fires
*before* the imminent window - a "fall predicted" 4 seconds before the person
even loses balance. By the project's own labelling that frame is a negative, so
scoring it as a successful anticipation would reward the model for contradicting
its ground truth, and would let a model that simply fires early on everything
post enormous lead times. The default here (`early_trigger="false_alarm"`)
counts it as what a caregiver would experience: a false alarm. `"hit"` and
`"ignore"` are available so the choice can be reported as a sensitivity check
rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.clips import ClipRecord
from data.labels import IGNORE_INDEX

from .decision import DecisionConfig, all_triggers

EARLY_TRIGGER_POLICIES = ("false_alarm", "hit", "ignore")


@dataclass
class ClipScores:
    """One clip's score stream, aligned frame-for-frame with the clip."""

    clip: ClipRecord
    scores: np.ndarray
    """`[T]` per-frame imminence probabilities."""

    labels: np.ndarray
    """`[T]` per-frame labels, possibly containing `IGNORE_INDEX`."""

    tti: np.ndarray
    """`[T]` seconds until impact."""

    def __post_init__(self) -> None:
        n = self.clip.num_frames
        for name in ("scores", "labels", "tti"):
            arr = np.asarray(getattr(self, name))
            if arr.shape[0] != n:
                raise ValueError(
                    f"{self.clip.clip_id}: {name} has {arr.shape[0]} frames, "
                    f"clip has {n}"
                )
            setattr(self, name, arr)


@dataclass
class ClipOutcome:
    """What the decision rule did on one clip."""

    clip_id: str
    is_fall: bool
    warned: bool
    """A valid in-window warning fired before impact."""

    lead_time: float | None
    """Seconds between the warning and impact, for a valid warning only."""

    trigger_frame: int | None
    false_alarms: int
    early_trigger: bool
    activity: str = "unknown"
    view: str = "default"


def _tied_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with tied values sharing their group's mean rank."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    n = ordered.size
    # First index of each run of equal values, and one past its last index.
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], n]
    group_mean = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(group_mean, ends - starts)
    return ranks


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank statistic, with ties averaged. NaN if one class is absent.

    Ties must be averaged rather than broken arbitrarily: a saturated model that
    outputs exactly 1.0 on many frames would otherwise score an AUC decided by
    array order rather than by anything it learned.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = _tied_ranks(np.concatenate([pos, neg]))
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed by the step-wise sum."""
    if (labels == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order] == 1
    cum_tp = np.cumsum(hits)
    precision = cum_tp / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / max(int(hits.sum()), 1))


def evaluate_clip(
    item: ClipScores,
    decision: DecisionConfig,
    w_pre: int,
    early_trigger: str = "false_alarm",
    tolerance_frames: int = 0,
) -> ClipOutcome:
    """Apply the decision rule to one clip and classify the result.

    Args:
        item: the clip and its score stream.
        decision: threshold, persistence and refractory period.
        w_pre: imminent-window length in frames, used to bound a valid warning.
        early_trigger: how to treat a trigger before the imminent window; one of
            `EARLY_TRIGGER_POLICIES`.
        tolerance_frames: grace period added before the imminent window, for the
            annotation uncertainty in `t*` (Section 18).
    """
    if early_trigger not in EARLY_TRIGGER_POLICIES:
        raise ValueError(
            f"early_trigger must be one of {EARLY_TRIGGER_POLICIES}, got {early_trigger!r}"
        )

    triggers = all_triggers(item.scores, decision)
    clip = item.clip

    if not clip.is_fall:
        return ClipOutcome(
            clip_id=clip.clip_id,
            is_fall=False,
            warned=False,
            lead_time=None,
            trigger_frame=triggers[0] if triggers else None,
            false_alarms=len(triggers),
            early_trigger=False,
            activity=clip.activity,
            view=clip.view,
        )

    impact = int(clip.impact_frame)
    window_start = impact - w_pre - tolerance_frames

    valid = [t for t in triggers if window_start <= t <= impact]
    early = [t for t in triggers if t < window_start]
    # Triggers after impact are post-hoc detections, not anticipations; they are
    # neither credited nor penalised.

    if valid:
        warn = min(valid)
        return ClipOutcome(
            clip_id=clip.clip_id,
            is_fall=True,
            warned=True,
            lead_time=float((impact - warn) / clip.fps),
            trigger_frame=warn,
            false_alarms=len(early) if early_trigger == "false_alarm" else 0,
            early_trigger=bool(early),
            activity=clip.activity,
            view=clip.view,
        )

    if early and early_trigger == "hit":
        warn = min(early)
        return ClipOutcome(
            clip_id=clip.clip_id,
            is_fall=True,
            warned=True,
            lead_time=float((impact - warn) / clip.fps),
            trigger_frame=warn,
            false_alarms=0,
            early_trigger=True,
            activity=clip.activity,
            view=clip.view,
        )

    return ClipOutcome(
        clip_id=clip.clip_id,
        is_fall=True,
        warned=False,
        lead_time=None,
        trigger_frame=early[0] if early else None,
        false_alarms=len(early) if early_trigger == "false_alarm" else 0,
        early_trigger=bool(early),
        activity=clip.activity,
        view=clip.view,
    )


@dataclass
class AnticipationReport:
    """The row this run contributes to the results table."""

    recall: float
    specificity: float
    false_alarms_per_hour: float
    mean_lead_time: float
    median_lead_time: float
    std_lead_time: float
    frame_auc: float
    frame_ap: float
    frame_f1: float
    frame_precision: float
    frame_recall: float
    num_falls: int
    num_adls: int
    num_warned: int
    num_false_alarms: int
    negative_hours: float
    decision: dict = field(default_factory=dict)
    outcomes: list[ClipOutcome] = field(default_factory=list)

    def to_row(self) -> dict:
        """Flat dict for CSV/JSON logging, without the per-clip detail."""
        row = {k: v for k, v in self.__dict__.items() if k != "outcomes"}
        row["decision"] = dict(self.decision)
        return row

    def summary(self) -> str:
        return (
            f"recall {self.recall:.3f} | lead {self.mean_lead_time:.3f}s "
            f"| FA/h {self.false_alarms_per_hour:.2f} | spec {self.specificity:.3f} "
            f"| frame AUC {self.frame_auc:.3f}"
        )


def evaluate(
    items: list[ClipScores],
    decision: DecisionConfig,
    w_pre: int,
    early_trigger: str = "false_alarm",
    tolerance_frames: int = 0,
) -> AnticipationReport:
    """Aggregate clip outcomes and frame-level scores into one report.

    The false-alarm rate is expressed per hour of **negative** time - frames
    genuinely labelled normal - rather than per hour of footage. Falls occupy a
    tiny fraction of any recording, so the two differ little in practice, but
    the negative-time denominator is the one that stays comparable when the
    fall/ADL ratio of the test set changes.
    """
    outcomes = [
        evaluate_clip(item, decision, w_pre, early_trigger, tolerance_frames)
        for item in items
    ]

    falls = [o for o in outcomes if o.is_fall]
    adls = [o for o in outcomes if not o.is_fall]
    warned = [o for o in falls if o.warned]
    leads = np.array([o.lead_time for o in warned], dtype=np.float64)

    negative_frames = sum(int((item.labels == 0).sum()) for item in items)
    fps = float(np.mean([item.clip.fps for item in items])) if items else 30.0
    negative_hours = negative_frames / fps / 3600.0
    total_false_alarms = sum(o.false_alarms for o in outcomes)

    all_scores = np.concatenate([item.scores for item in items]) if items else np.zeros(0)
    all_labels = np.concatenate([item.labels for item in items]) if items else np.zeros(0)
    keep = all_labels != IGNORE_INDEX
    scores, labels = all_scores[keep], all_labels[keep]

    predicted = scores >= decision.threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    frame_recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * frame_recall / max(precision + frame_recall, 1e-9)

    return AnticipationReport(
        recall=len(warned) / max(len(falls), 1),
        specificity=sum(1 for o in adls if o.false_alarms == 0) / max(len(adls), 1),
        false_alarms_per_hour=total_false_alarms / max(negative_hours, 1e-9),
        mean_lead_time=float(leads.mean()) if leads.size else float("nan"),
        median_lead_time=float(np.median(leads)) if leads.size else float("nan"),
        std_lead_time=float(leads.std(ddof=0)) if leads.size else float("nan"),
        frame_auc=_auc_from_scores(scores, labels),
        frame_ap=_average_precision(scores, labels),
        frame_f1=f1,
        frame_precision=precision,
        frame_recall=frame_recall,
        num_falls=len(falls),
        num_adls=len(adls),
        num_warned=len(warned),
        num_false_alarms=total_false_alarms,
        negative_hours=negative_hours,
        decision={
            "threshold": decision.threshold,
            "persistence": decision.persistence,
            "refractory_frames": decision.refractory_frames,
            "early_trigger": early_trigger,
        },
        outcomes=outcomes,
    )


def operating_curve(
    items: list[ClipScores],
    configs: list[DecisionConfig],
    w_pre: int,
    early_trigger: str = "false_alarm",
) -> list[dict]:
    """Section 10's operating-point curve: one row per `(tau, k)` pair.

    Reported instead of a single threshold, because a single threshold is always
    open to the charge of having been chosen after seeing the test set.
    """
    return [
        evaluate(items, cfg, w_pre, early_trigger).to_row() for cfg in configs
    ]


def select_operating_point(
    rows: list[dict],
    max_false_alarms_per_hour: float = 1.0,
    objective: str = "mean_lead_time",
) -> dict | None:
    """Pick the operating point on the **validation** set, never the test set.

    Maximises `objective` subject to a false-alarm budget, and breaks ties on
    recall. Returns None when no point meets the budget, which is itself worth
    reporting rather than papering over by relaxing the budget.
    """
    feasible = [
        r for r in rows
        if r["false_alarms_per_hour"] <= max_false_alarms_per_hour and r["num_warned"] > 0
    ]
    if not feasible:
        return None

    def key(row: dict):
        value = row.get(objective, float("nan"))
        return (0.0 if np.isnan(value) else value, row["recall"])

    return max(feasible, key=key)
