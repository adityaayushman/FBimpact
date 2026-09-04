"""Per-frame output heads.

The binary head is the original: one imminence logit per frame, consumed by the
threshold-and-persistence rule.

`MultiTaskHead` adds a four-way phase classifier alongside it, sharing the whole
backbone. Two things make that worth the extra parameters:

**The phase target is denser.** Binary supervision says "positive" over the
imminent window and masks everything after impact. Phases label every frame,
including the grounded ones, and distinguish *about to lose balance* from
*already going down*. More labelled frames per clip is more gradient signal from
the same data - which matters when the whole real benchmark is 129 falls.

**The two heads must agree.** The imminence logit is not independent of the
phase distribution: `imminent + falling` is by construction the binary positive
class. Predicting them separately and letting them disagree would be wasteful,
so `consistency_loss` penalises the gap between the binary head's probability
and the phase head's pre-impact mass. It is a regulariser, not a constraint -
the binary head stays free to be the one the decision rule reads, so lead time
is still measured from exactly the quantity it was measured from before.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.phases import NUM_PHASES, PRE_IMPACT_PHASES


class BinaryHead(nn.Module):
    """One fall-imminence logit per frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """`[N, C, T]` -> `[N, T]`."""
        return self.head(pooled).squeeze(1)


class MultiTaskHead(nn.Module):
    """Binary imminence plus a four-way phase classifier."""

    def __init__(self, channels: int, num_phases: int = NUM_PHASES) -> None:
        super().__init__()
        self.binary = nn.Conv1d(channels, 1, kernel_size=1)
        self.phase = nn.Conv1d(channels, num_phases, kernel_size=1)
        self.num_phases = num_phases

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled: `[N, C, T]` joint-pooled features.

        Returns:
            `(imminence [N, T], phase logits [N, num_phases, T])`.
        """
        return self.binary(pooled).squeeze(1), self.phase(pooled)


def phase_to_imminence(phase_logits: torch.Tensor) -> torch.Tensor:
    """Probability mass on the phases that precede ground contact. `[N, T]`.

    The phase head's own answer to "is a fall imminent", used for the
    consistency term and available as an alternative score at inference.
    """
    probabilities = torch.softmax(phase_logits, dim=1)
    return probabilities[:, list(PRE_IMPACT_PHASES), :].sum(dim=1)


def consistency_loss(
    binary_logits: torch.Tensor,
    phase_logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Squared gap between the two heads' imminence estimates.

    Applied only where the frame is valid, and detached from neither side: both
    heads are pulled toward agreement rather than one being fitted to the other,
    which would make the phase head a follower and waste its denser supervision.
    """
    if not bool(valid.any()):
        return binary_logits.sum() * 0.0
    binary = torch.sigmoid(binary_logits)
    from_phase = phase_to_imminence(phase_logits)
    return (F.mse_loss(binary, from_phase, reduction="none") * valid).sum() / valid.sum()
