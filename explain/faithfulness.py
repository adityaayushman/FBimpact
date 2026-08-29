"""Stage F - is the explanation faithful, or merely plausible? (RQ2)

Deletion/insertion after Petsiuk et al., RISE (BMVC 2018), adapted from image
pixels to skeleton joints.

* **Deletion** removes joints in decreasing order of relevance and records the
  score at the warning frame. If the ranking is faithful the score collapses
  early, so the area under the curve is *low*.
* **Insertion** starts from a blank skeleton and adds joints in the same order.
  A faithful ranking recovers the score early, so the area is *high*.

Neither number means anything on its own: a model whose score barely depends on
any single joint produces a flat curve and a middling area under both. What is
reported is the **gap against a random joint ordering** on the same clips and the
same model - `deletion_gap` and `insertion_gap`, both signed so that positive is
better. A gap of zero is the honest negative result Section 18 anticipates:
the attention looked reasonable and explained nothing.

A caveat that belongs in the paper rather than a footnote: removing a joint
produces a skeleton the model never saw in training, so part of any score drop
is distribution shift rather than lost evidence. That is why the random baseline
is subtracted (it suffers the same shift) and why `baseline="neighbour"`, which
replaces a joint with the mean of its graph neighbours and keeps the skeleton
anatomically coherent, is offered as a check on the cruder `"zero"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from data.skeleton import NUM_JOINTS, adjacency_matrix

BASELINES = ("zero", "mean", "neighbour")

# Cached neighbour-averaging operator: row i averages joint i's graph neighbours.
_NEIGHBOUR_OP: np.ndarray | None = None


def _neighbour_operator() -> np.ndarray:
    global _NEIGHBOUR_OP
    if _NEIGHBOUR_OP is None:
        a = adjacency_matrix(self_loops=False)
        degree = a.sum(axis=1, keepdims=True)
        _NEIGHBOUR_OP = np.divide(a, degree, out=np.zeros_like(a), where=degree > 0)
    return _NEIGHBOUR_OP


def apply_baseline(
    window: torch.Tensor,
    joints,
    baseline: str = "zero",
    mean_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a copy of `window` with `joints` replaced by the baseline value.

    Args:
        window: `[1, C, T, V, M]`.
        joints: iterable of joint indices to remove.
        baseline: `"zero"` puts the joint at the body centre with zero velocity;
            `"mean"` uses `mean_pose`; `"neighbour"` uses the mean of the
            joint's graph neighbours, which keeps the skeleton plausible.
        mean_pose: `[C, V]` dataset mean, required for `baseline="mean"`.
    """
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
    joints = list(joints)
    out = window.clone()
    if not joints:
        return out

    if baseline == "zero":
        out[:, :, :, joints, :] = 0.0
        return out

    if baseline == "mean":
        if mean_pose is None:
            raise ValueError("baseline='mean' needs a mean_pose of shape [C, V]")
        mean_pose = mean_pose.to(out.device, out.dtype)
        for j in joints:
            out[:, :, :, j, :] = mean_pose[:, j].view(1, -1, 1, 1)
        return out

    # neighbour: average the *original* neighbours, so the order of removal
    # cannot cascade one imputation into the next.
    op = torch.from_numpy(_neighbour_operator()).to(out.device, out.dtype)
    for j in joints:
        # [N, C, T, V, M] weighted over V by joint j's neighbours -> [N, C, T, M]
        out[:, :, :, j, :] = torch.einsum("nctvm,v->nctm", window, op[j])
    return out


@dataclass
class FaithfulnessCurves:
    """Deletion and insertion curves for one explanation method."""

    deletion: np.ndarray
    """`[V+1]` score after removing 0..V joints in relevance order."""

    insertion: np.ndarray
    """`[V+1]` score after adding 0..V joints in relevance order."""

    deletion_random: np.ndarray
    insertion_random: np.ndarray
    method: str = "attention"

    @staticmethod
    def _auc(curve: np.ndarray) -> float:
        """Normalised area under a curve sampled at equal steps.

        Divided by the number of steps so the value stays on the same 0-1 scale
        as the scores themselves and stays comparable across skeleton layouts
        with different joint counts.
        """
        curve = np.asarray(curve, dtype=np.float64)
        steps = max(len(curve) - 1, 1)
        area = float(curve.sum() - 0.5 * (curve[0] + curve[-1]))  # trapezoid rule
        return area / steps

    @property
    def deletion_auc(self) -> float:
        return self._auc(self.deletion)

    @property
    def insertion_auc(self) -> float:
        return self._auc(self.insertion)

    @property
    def deletion_gap(self) -> float:
        """`random - ours`; positive means our ranking destroys the score faster."""
        return self._auc(self.deletion_random) - self.deletion_auc

    @property
    def insertion_gap(self) -> float:
        """`ours - random`; positive means our ranking restores the score faster."""
        return self.insertion_auc - self._auc(self.insertion_random)


@dataclass
class FaithfulnessReport:
    """Aggregate over the evaluated warnings - the number RQ2 answers with."""

    method: str
    deletion_auc: float
    insertion_auc: float
    deletion_auc_random: float
    insertion_auc_random: float
    deletion_gap: float
    insertion_gap: float
    deletion_gap_std: float
    insertion_gap_std: float
    num_explanations: int
    baseline: str
    curves: list[FaithfulnessCurves] = field(default_factory=list)

    def to_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "curves"}

    def summary(self) -> str:
        return (
            f"{self.method}: deletion {self.deletion_auc:.3f} "
            f"(random {self.deletion_auc_random:.3f}, gap {self.deletion_gap:+.3f}) | "
            f"insertion {self.insertion_auc:.3f} "
            f"(random {self.insertion_auc_random:.3f}, gap {self.insertion_gap:+.3f}) "
            f"over {self.num_explanations} warnings"
        )


@torch.no_grad()
def _curve(
    model: torch.nn.Module,
    window: torch.Tensor,
    frame: int,
    order: np.ndarray,
    mode: str,
    baseline: str,
    mean_pose: torch.Tensor | None,
) -> np.ndarray:
    """Score at `frame` after progressively removing (or adding) joints."""
    scores = np.empty(NUM_JOINTS + 1, dtype=np.float64)
    for step in range(NUM_JOINTS + 1):
        if mode == "deletion":
            removed = order[:step]
        else:  # insertion: everything except the first `step` joints is removed
            removed = order[step:]
        perturbed = apply_baseline(window, removed, baseline, mean_pose)
        scores[step] = torch.sigmoid(model(perturbed))[0, frame].item()
    return scores


def faithfulness_curves(
    model: torch.nn.Module,
    window: torch.Tensor,
    frame: int,
    relevance: np.ndarray,
    device: torch.device,
    baseline: str = "zero",
    num_random: int = 5,
    seed: int = 0,
    mean_pose: torch.Tensor | None = None,
    method: str = "attention",
) -> FaithfulnessCurves:
    """Deletion and insertion curves for one warning, with a random control.

    Args:
        model: the trained model, in eval mode.
        window: `[C, T, V]` or `[1, C, T, V, M]` window containing the warning.
        frame: position within the window to explain.
        relevance: `[V]` joint relevance from Stage E.
        device: where to run.
        baseline: how a removed joint is filled; see `apply_baseline`.
        num_random: random orderings averaged for the control curve. More is
            steadier but costs `num_random x (V+1)` extra forward passes.
        seed: seed for the random orderings, so the control is reproducible.
        mean_pose: `[C, V]`, required when `baseline="mean"`.
        method: recorded on the result for reporting.
    """
    from .relevance import _prepare_window

    window = _prepare_window(window, device)
    window_len = window.shape[2]
    frame = int(frame) % window_len
    order = np.argsort(-np.asarray(relevance), kind="mergesort")

    was_training = model.training
    model.eval()

    deletion = _curve(model, window, frame, order, "deletion", baseline, mean_pose)
    insertion = _curve(model, window, frame, order, "insertion", baseline, mean_pose)

    rng = np.random.default_rng(seed)
    random_deletion = np.zeros_like(deletion)
    random_insertion = np.zeros_like(insertion)
    for _ in range(max(num_random, 1)):
        shuffled = rng.permutation(NUM_JOINTS)
        random_deletion += _curve(
            model, window, frame, shuffled, "deletion", baseline, mean_pose
        )
        random_insertion += _curve(
            model, window, frame, shuffled, "insertion", baseline, mean_pose
        )
    random_deletion /= max(num_random, 1)
    random_insertion /= max(num_random, 1)

    if was_training:
        model.train()

    return FaithfulnessCurves(
        deletion=deletion,
        insertion=insertion,
        deletion_random=random_deletion,
        insertion_random=random_insertion,
        method=method,
    )


def aggregate(curves: list[FaithfulnessCurves], baseline: str = "zero") -> FaithfulnessReport:
    """Average per-warning curves into the reported faithfulness result.

    The standard deviations are over warnings, not over seeds, and answer a
    different question from the seed variance in Section 14: they say whether
    the explanation is consistently faithful or faithful only on average.
    """
    if not curves:
        raise ValueError("no curves to aggregate - were any warnings produced?")

    deletion_gaps = np.array([c.deletion_gap for c in curves])
    insertion_gaps = np.array([c.insertion_gap for c in curves])

    return FaithfulnessReport(
        method=curves[0].method,
        deletion_auc=float(np.mean([c.deletion_auc for c in curves])),
        insertion_auc=float(np.mean([c.insertion_auc for c in curves])),
        deletion_auc_random=float(
            np.mean([FaithfulnessCurves._auc(c.deletion_random) for c in curves])
        ),
        insertion_auc_random=float(
            np.mean([FaithfulnessCurves._auc(c.insertion_random) for c in curves])
        ),
        deletion_gap=float(deletion_gaps.mean()),
        insertion_gap=float(insertion_gaps.mean()),
        deletion_gap_std=float(deletion_gaps.std(ddof=0)),
        insertion_gap_std=float(insertion_gaps.std(ddof=0)),
        num_explanations=len(curves),
        baseline=baseline,
        curves=curves,
    )
