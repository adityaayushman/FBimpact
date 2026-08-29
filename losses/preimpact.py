"""Section 14 - the pre-impact, time-weighted objective.

Plain per-frame BCE treats every frame of the imminent window as equally
valuable. It is not: a correct warning 0.6 s before impact is worth far more than
the same warning 0.05 s before it, because only the early one leaves time for a
protective response. This loss makes that explicit by scaling each positive
frame's contribution by

    w(d) = exp(lambda * d) / Z(lambda, D),    d = seconds until impact

so that earlier frames inside the imminent window carry more weight, and `Z`
normalises the average weight over the window to 1 so `lambda` can be changed
without also changing the effective learning rate.

Setting `lambda = 0` recovers plain class-weighted BCE, which is exactly the
`- pre-impact loss` ablation from Section 16: the same architecture, the same
data, the same schedule, differing only in this one scalar. That is what makes
RQ3 - does the pre-impact objective buy lead time, and at what cost in recall or
false alarms - answerable rather than merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.labels import IGNORE_INDEX


@dataclass
class PreImpactLossConfig:
    """Parameters of the anticipation objective."""

    lam: float = 1.5
    """Time-weighting strength, per second of lead time. 0 disables it."""

    w_pre_seconds: float = 0.67
    """Length of the imminent window in seconds; sets the normaliser `Z`."""

    pos_weight: float | None = None
    """Positive-class weight. None derives it from the training set's positive
    fraction (Section 14: class weighting for the fall/normal imbalance)."""

    max_pos_weight: float = 20.0
    """Ceiling on a derived `pos_weight`. Without it a rare-positive split can
    produce a weight in the hundreds, which destabilises training long before it
    improves recall."""

    focal_gamma: float = 0.0
    """Focal-loss exponent. 0 is plain BCE; ~2 down-weights easy frames."""

    label_smoothing: float = 0.0
    """Pulls targets towards 0.5. A little smoothing helps because the exact
    frame at which a fall becomes 'imminent' is itself an annotation judgement."""


def time_weights(tti: torch.Tensor, lam: float, w_pre_seconds: float) -> torch.Tensor:
    """Per-frame weights `w(d)`, normalised to mean 1 over the imminent window.

    Args:
        tti: `[N, T]` seconds until impact - positive before impact, negative
            after, `+inf` for ADL clips.
        lam: time-weighting strength.
        w_pre_seconds: length of the imminent window in seconds.

    Returns:
        `[N, T]` weights. Frames outside the imminent window get weight 1, so
        the negatives keep their natural scale.
    """
    if lam == 0.0:
        return torch.ones_like(tti)

    horizon = max(float(w_pre_seconds), 1e-6)
    # Only frames inside [0, horizon] are re-weighted; the rest stay at 1.
    inside = torch.isfinite(tti) & (tti >= 0.0) & (tti <= horizon)
    d = tti.clamp(min=0.0, max=horizon)

    # Z = mean of exp(lam * d) for d uniform on [0, horizon], so E[w] = 1.
    lh = lam * horizon
    z = torch.expm1(torch.tensor(lh, device=tti.device, dtype=tti.dtype)) / lh
    weights = torch.exp(lam * d) / z
    return torch.where(inside, weights, torch.ones_like(weights))


class PreImpactLoss(nn.Module):
    """Masked, class-weighted, time-weighted binary cross-entropy over frames."""

    def __init__(self, config: PreImpactLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or PreImpactLossConfig()

    def set_pos_weight_from_prior(self, positive_fraction: float) -> float:
        """Derive `pos_weight` from the training set's positive frame fraction.

        Uses `(1 - p) / p`, the weight that equalises the two classes' total
        contribution, clipped at `max_pos_weight`. Returns the value used, so
        the run log can record it.
        """
        p = min(max(float(positive_fraction), 1e-6), 1.0 - 1e-6)
        weight = min((1.0 - p) / p, self.config.max_pos_weight)
        self.config.pos_weight = float(weight)
        return self.config.pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tti: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits: `[N, T]` raw per-frame outputs.
            labels: `[N, T]` of 0, 1 or `IGNORE_INDEX`.
            tti: `[N, T]` seconds until impact.

        Returns:
            `(loss, stats)`. `stats` carries the pieces worth logging: how much
            of the batch was masked out, and the positive/negative split of the
            loss, which is the first thing to look at when recall collapses.
        """
        cfg = self.config
        valid = labels != IGNORE_INDEX
        if not bool(valid.any()):
            # A batch of entirely post-impact frames is possible but rare; return
            # a real zero that still participates in the graph.
            return logits.sum() * 0.0, {"valid_fraction": 0.0}

        target = labels.clamp(min=0).to(logits.dtype)
        if cfg.label_smoothing > 0.0:
            s = cfg.label_smoothing
            target = target * (1.0 - s) + 0.5 * s

        pos_weight = (
            torch.tensor(cfg.pos_weight, device=logits.device, dtype=logits.dtype)
            if cfg.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )

        if cfg.focal_gamma > 0.0:
            # p_t is the probability assigned to the *correct* class.
            p = torch.sigmoid(logits)
            p_t = p * target + (1.0 - p) * (1.0 - target)
            bce = bce * (1.0 - p_t).clamp(min=0.0) ** cfg.focal_gamma

        weights = time_weights(tti, cfg.lam, cfg.w_pre_seconds)
        weighted = bce * weights * valid

        denominator = (weights * valid).sum().clamp(min=1e-8)
        loss = weighted.sum() / denominator

        with torch.no_grad():
            is_pos = valid & (labels == 1)
            is_neg = valid & (labels == 0)
            stats = {
                "valid_fraction": valid.float().mean().item(),
                "positive_fraction": is_pos.float().sum().item()
                / max(valid.float().sum().item(), 1.0),
                "loss_positive": (bce * is_pos).sum().item() / max(is_pos.sum().item(), 1),
                "loss_negative": (bce * is_neg).sum().item() / max(is_neg.sum().item(), 1),
                "mean_time_weight": (weights * is_pos).sum().item()
                / max(is_pos.sum().item(), 1),
            }
        return loss, stats


def build_loss(cfg: dict | None) -> PreImpactLoss:
    """Build the loss from the `loss` block of a run config."""
    cfg = dict(cfg or {})
    fields = set(PreImpactLossConfig.__dataclass_fields__)
    unknown = set(cfg) - fields
    if unknown:
        raise ValueError(f"unknown loss options: {sorted(unknown)}; valid: {sorted(fields)}")
    return PreImpactLoss(PreImpactLossConfig(**cfg))
