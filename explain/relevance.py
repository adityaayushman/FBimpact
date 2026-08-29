"""Stage E - which joints drove this warning.

Two relevance sources, deliberately kept separate rather than blended:

* **Attention** reads the weights the model's own pooling used. It is free, it
  is exactly the quantity Section 7 promises to expose, and it is available at
  inference with no extra pass.
* **Gradient x input** measures how much the logit at the warning frame actually
  moves with each joint. It is more expensive and needs a backward pass, but it
  is sensitive to information that bypasses the attention pooling - a joint can
  matter through the graph convolutions while carrying a low attention weight.

Reporting both matters because the two disagreeing is informative: attention that
looks convincing while the gradients point elsewhere is the "plausible but
unfaithful" failure RQ2 exists to catch. Neither is trusted on its own; Stage F
decides which, if either, is faithful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data.skeleton import JOINT_NAMES, NUM_JOINTS

METHODS = ("attention", "gradient", "gradient_input", "occlusion")

# Phrases a caregiver-facing message can use, keyed by the joint group that
# dominates the top-k. Deliberately descriptive of kinematics, never diagnostic.
_GROUP_PHRASES = {
    "trunk": "trunk lean",
    "hips": "hip drop",
    "knees": "knee buckling",
    "ankles": "foot instability",
    "arms": "arm bracing",
    "head": "head pitch",
}

_GROUP_MEMBERS = {
    "head": (0, 1, 2, 3, 4),
    "trunk": (5, 6),
    "arms": (7, 8, 9, 10),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}


@dataclass
class JointRelevance:
    """A ranked joint attribution for one (clip, frame) pair."""

    frame: int
    """Frame within the clip that the explanation refers to."""

    scores: np.ndarray
    """`[V]` relevance, non-negative and summing to 1."""

    method: str

    @property
    def order(self) -> np.ndarray:
        """Joint indices, most relevant first."""
        return np.argsort(-self.scores, kind="mergesort")

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """The `k` most relevant joints as `(name, score)`."""
        return [(JOINT_NAMES[i], float(self.scores[i])) for i in self.order[:k]]

    def phrase(self, k: int = 3) -> str:
        """A short human-readable summary, e.g. "trunk lean, hip drop"."""
        groups: list[str] = []
        for joint in self.order[:k]:
            for name, members in _GROUP_MEMBERS.items():
                if joint in members and name not in groups:
                    groups.append(name)
                    break
        return ", ".join(_GROUP_PHRASES.get(g, g) for g in groups) or "no dominant joint"


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Clip to non-negative and scale to sum 1, so methods stay comparable."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 0.0, None)
    total = scores.sum()
    if total <= 0.0:
        return np.full(scores.shape, 1.0 / scores.size)
    return scores / total


def _prepare_window(window: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Accept `[C,T,V]`, `[1,C,T,V]` or `[1,C,T,V,1]` and return `[1,C,T,V,1]`."""
    if window.dim() == 3:
        window = window.unsqueeze(0)
    if window.dim() == 4:
        window = window.unsqueeze(-1)
    if window.dim() != 5 or window.shape[0] != 1:
        raise ValueError(
            f"expected a single window of shape [C,T,V], got {tuple(window.shape)}"
        )
    return window.to(device)


@torch.no_grad()
def attention_relevance(
    model: torch.nn.Module, window: torch.Tensor, frame: int, device: torch.device
) -> np.ndarray:
    """Joint attention weights at `frame` (a position inside the window)."""
    window = _prepare_window(window, device)
    _, aux = model(window, return_aux=True)
    weights = aux["attention"][0]                      # [T, V]
    return _normalise(weights[frame].detach().cpu().numpy())


def gradient_relevance(
    model: torch.nn.Module,
    window: torch.Tensor,
    frame: int,
    device: torch.device,
    times_input: bool = True,
) -> np.ndarray:
    """Sensitivity of the logit at `frame` to each joint's trajectory.

    The gradient is taken with respect to the whole window and then summed over
    channels and over time, because a joint influences the warning through its
    entire recent trajectory, not just its position on the warning frame. With
    `times_input`, gradients are multiplied by the input first, which measures
    contribution rather than pure sensitivity.
    """
    window = _prepare_window(window, device).clone().requires_grad_(True)
    was_training = model.training
    model.eval()

    logits = model(window)
    model.zero_grad(set_to_none=True)
    logits[0, frame].backward()

    grad = window.grad
    if grad is None:
        raise RuntimeError("no gradient reached the input; is the model detached?")
    contribution = grad * window if times_input else grad
    # [1, C, T, V, 1] -> [V]
    per_joint = contribution.detach().abs().sum(dim=(0, 1, 2, 4)).cpu().numpy()

    if was_training:
        model.train()
    return _normalise(per_joint)


@torch.no_grad()
def occlusion_relevance(
    model: torch.nn.Module,
    window: torch.Tensor,
    frame: int,
    device: torch.device,
    baseline: str = "zero",
) -> np.ndarray:
    """Drop in the score at `frame` when each joint alone is removed.

    The most literal notion of "this joint mattered", and the most expensive:
    one forward pass per joint. Useful as a reference that the cheaper methods
    can be checked against, since the faithfulness test perturbs joints in
    exactly this way.
    """
    from .faithfulness import apply_baseline

    window = _prepare_window(window, device)
    reference = torch.sigmoid(model(window))[0, frame].item()

    drops = np.zeros(NUM_JOINTS, dtype=np.float64)
    for joint in range(NUM_JOINTS):
        perturbed = apply_baseline(window, [joint], baseline)
        drops[joint] = reference - torch.sigmoid(model(perturbed))[0, frame].item()
    return _normalise(drops)


def joint_relevance(
    model: torch.nn.Module,
    window: torch.Tensor,
    frame: int,
    device: torch.device,
    method: str = "attention",
) -> JointRelevance:
    """Compute joint relevance by the named method.

    Args:
        model: the trained model.
        window: `[C, T, V]` window whose last frames precede the warning.
        frame: position **within the window** to explain; use `-1` or `T-1` for
            the warning frame itself.
        device: where to run.
        method: one of `METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    window_len = window.shape[-2] if window.dim() >= 4 else window.shape[1]
    frame = int(frame) % window_len

    if method == "attention":
        scores = attention_relevance(model, window, frame, device)
    elif method in ("gradient", "gradient_input"):
        scores = gradient_relevance(
            model, window, frame, device, times_input=(method == "gradient_input")
        )
    else:
        scores = occlusion_relevance(model, window, frame, device)

    return JointRelevance(frame=frame, scores=scores, method=method)
