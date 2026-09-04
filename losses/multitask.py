"""Binary imminence + four-phase classification, optionally tied together.

The binary term is the existing `PreImpactLoss` unchanged, so the quantity the
decision rule reads and lead time is measured from stays exactly what it was.
Everything here is added supervision on top of it, never a replacement:

    total = binary + phase_weight * phase + consistency_weight * consistency

`phase_weight = 0` recovers the original objective bit for bit, which is what
makes the phase head an ablatable component rather than a rewrite. That matters
given the project's headline finding is that its previous "improvement" made
things worse - a new component that cannot be switched off cleanly cannot be
shown to help either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.labels import IGNORE_INDEX
from data.phases import NUM_PHASES

from .preimpact import PreImpactLoss, PreImpactLossConfig


@dataclass
class MultiTaskLossConfig:
    """Weights for the combined objective."""

    binary: PreImpactLossConfig = field(default_factory=PreImpactLossConfig)

    phase_weight: float = 0.5
    """Strength of the phase term. 0 disables it exactly."""

    consistency_weight: float = 0.1
    """Pulls the binary head and the phase head's pre-impact mass together.
    Small on purpose: it is a regulariser, and a large value would make one head
    fit the other rather than both fit the data."""

    phase_class_weights: list[float] | None = None
    """Per-phase weights. None derives inverse-frequency weights from the
    training set, capped, exactly as the binary term caps `pos_weight`."""

    label_smoothing: float = 0.0


class MultiTaskLoss(nn.Module):
    """Combined imminence and phase objective."""

    def __init__(self, config: MultiTaskLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiTaskLossConfig()
        self.binary_loss = PreImpactLoss(self.config.binary)
        self._weights: torch.Tensor | None = None
        if self.config.phase_class_weights is not None:
            self._weights = torch.tensor(self.config.phase_class_weights, dtype=torch.float32)

    def set_phase_weights_from_distribution(self, distribution: dict[str, float]) -> list[float]:
        """Derive capped inverse-frequency phase weights; returns them for logging."""
        from data.phases import class_weights

        weights = class_weights(distribution)
        self._weights = torch.tensor(weights, dtype=torch.float32)
        self.config.phase_class_weights = [round(float(w), 4) for w in weights]
        return self.config.phase_class_weights

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tti: torch.Tensor,
        phase_logits: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits: `[N, T]` binary imminence logits.
            labels: `[N, T]` binary targets, possibly `IGNORE_INDEX`.
            tti: `[N, T]` seconds to impact.
            phase_logits: `[N, P, T]` phase logits, or None for a binary-only model.
            phases: `[N, T]` phase targets, possibly `IGNORE_INDEX`.

        Returns:
            `(loss, stats)`.
        """
        total, stats = self.binary_loss(logits, labels, tti)
        stats = {f"binary_{k}": v for k, v in stats.items()}

        cfg = self.config
        if phase_logits is None or phases is None or cfg.phase_weight <= 0.0:
            return total, stats

        valid = phases != IGNORE_INDEX
        if not bool(valid.any()):
            return total, stats

        weights = None
        if self._weights is not None:
            weights = self._weights.to(logits.device, logits.dtype)

        # cross_entropy ignores IGNORE_INDEX directly, so masked frames cost
        # nothing rather than being clamped into a class they do not belong to.
        phase_loss = F.cross_entropy(
            phase_logits, phases.clamp(min=IGNORE_INDEX),
            weight=weights, ignore_index=IGNORE_INDEX,
            label_smoothing=cfg.label_smoothing,
        )
        total = total + cfg.phase_weight * phase_loss
        stats["phase_loss"] = float(phase_loss.item())

        if cfg.consistency_weight > 0.0:
            from models.heads import consistency_loss

            agree = consistency_loss(logits, phase_logits, valid.to(logits.dtype))
            total = total + cfg.consistency_weight * agree
            stats["consistency"] = float(agree.item())

        with torch.no_grad():
            predicted = phase_logits.argmax(dim=1)
            correct = ((predicted == phases) & valid).sum().item()
            stats["phase_accuracy"] = correct / max(int(valid.sum().item()), 1)
            for index in range(NUM_PHASES):
                is_class = valid & (phases == index)
                count = int(is_class.sum().item())
                if count:
                    stats[f"phase_recall_{index}"] = (
                        ((predicted == index) & is_class).sum().item() / count
                    )
        return total, stats


def build_multitask_loss(cfg: dict | None) -> MultiTaskLoss:
    """Build from the `loss` block of a run config.

    Keys belonging to the binary term are passed through to it, so an existing
    config that knows nothing about phases produces the identical objective.
    """
    cfg = dict(cfg or {})
    multi_keys = {"phase_weight", "consistency_weight", "phase_class_weights"}
    binary_fields = set(PreImpactLossConfig.__dataclass_fields__)

    binary_cfg = {k: v for k, v in cfg.items() if k in binary_fields}
    unknown = set(cfg) - binary_fields - multi_keys - {"label_smoothing"}
    if unknown:
        raise ValueError(f"unknown loss options: {sorted(unknown)}")

    return MultiTaskLoss(MultiTaskLossConfig(
        binary=PreImpactLossConfig(**binary_cfg),
        phase_weight=float(cfg.get("phase_weight", 0.5)),
        consistency_weight=float(cfg.get("consistency_weight", 0.1)),
        phase_class_weights=cfg.get("phase_class_weights"),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    ))
