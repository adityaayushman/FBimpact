"""Four-phase labelling: normal, imminent, falling, grounded.

The binary scheme in `data.labels` asks one question - is impact within `W_pre`
frames? - and throws away something both datasets actually annotate: **when the
fall visibly begins**. UR Fall marks a transition state between upright and
lying; Le2i annotates a fall start frame as well as an end. Collapsing those to
a single positive class discards the distinction between *about to lose balance*
and *already going down*, which is precisely the distinction an anticipation
model needs to learn.

    NORMAL     ordinary activity, or a fall clip well before onset
    IMMINENT   the `W_pre` frames leading up to onset - balance is about to go
    FALLING    onset .. impact - the body is on its way to the floor
    GROUNDED   after impact - already down, nothing left to anticipate

Two properties are worth stating because they decide whether this helps:

**It is strictly more information, never less.** The binary target is recovered
exactly as `IMMINENT or FALLING`, so a model trained on phases can still be
evaluated by the identical decision rule and metrics. Nothing in the reported
lead-time pipeline changes.

**GROUNDED replaces masking.** The binary scheme masks post-impact frames out of
the loss because they are neither imminent nor normal. Here they are a class in
their own right, which is both more honest and free supervision - "the person is
already down" is a genuinely different state, and a model that can name it is
less likely to confuse lying-down-deliberately with falling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMAL, IMMINENT, FALLING, GROUNDED = 0, 1, 2, 3

PHASE_NAMES = ("normal", "imminent", "falling", "grounded")
NUM_PHASES = len(PHASE_NAMES)

# Phases that precede ground contact. The binary imminence score a warning is
# fired from is the total probability mass over these two.
PRE_IMPACT_PHASES = (IMMINENT, FALLING)


@dataclass(frozen=True)
class PhaseConfig:
    """Parameters of the phase labelling."""

    w_pre: int = 20
    """Frames before onset that count as imminent."""

    include_grounded: bool = True
    """When False, post-impact frames are masked exactly as the binary scheme
    masks them, so the two labellings differ only in splitting the positive
    class. Useful for isolating which of the two changes is responsible for a
    difference in results."""


def phase_labels(
    num_frames: int,
    impact_frame: int | None,
    onset_frame: int | None = None,
    config: PhaseConfig | None = None,
) -> np.ndarray:
    """Per-frame phase labels for one clip.

    Args:
        num_frames: clip length `T`.
        impact_frame: `t*`, first ground contact, or None for an ADL clip.
        onset_frame: first frame of the visible fall. When absent, the whole
            imminent window before `t*` is labelled IMMINENT and no frame is
            labelled FALLING - the clip contributes to the other three classes
            and simply says nothing about this one.
        config: labelling parameters.

    Returns:
        `[T]` int array of phase indices, or `IGNORE_INDEX` where masked.
    """
    from .labels import IGNORE_INDEX

    config = config or PhaseConfig()
    labels = np.full(num_frames, NORMAL, dtype=np.int64)
    if impact_frame is None:
        return labels

    if not 0 <= impact_frame < num_frames:
        raise ValueError(f"impact_frame {impact_frame} outside {num_frames} frames")

    if onset_frame is None:
        # No annotated onset: treat the whole pre-impact window as imminent.
        # This is the binary scheme's positive class, kept rather than guessed
        # at a split point that the dataset does not provide.
        start = max(0, impact_frame - config.w_pre)
        labels[start : impact_frame + 1] = IMMINENT
    else:
        onset = int(np.clip(onset_frame, 0, impact_frame))
        labels[onset : impact_frame + 1] = FALLING
        labels[max(0, onset - config.w_pre) : onset] = IMMINENT

    if config.include_grounded:
        labels[impact_frame + 1 :] = GROUNDED
    else:
        labels[impact_frame + 1 :] = IGNORE_INDEX

    return labels


def to_binary(phases: np.ndarray) -> np.ndarray:
    """Collapse phases to the binary imminent/normal target.

    Exactly reproduces `data.labels.frame_labels` given the same `w_pre`, so
    phase-trained models stay comparable with binary-trained ones under one
    evaluation path. GROUNDED maps back to the ignore index it came from.
    """
    from .labels import IGNORE_INDEX

    binary = np.zeros_like(phases)
    binary[phases == IGNORE_INDEX] = IGNORE_INDEX
    binary[phases == GROUNDED] = IGNORE_INDEX
    binary[np.isin(phases, PRE_IMPACT_PHASES)] = 1
    return binary


def phase_distribution(phases_per_clip: list[np.ndarray]) -> dict[str, float]:
    """Fraction of frames in each phase, for class weighting and for logging."""
    from .labels import IGNORE_INDEX

    counts = np.zeros(NUM_PHASES, dtype=np.int64)
    for labels in phases_per_clip:
        kept = labels[labels != IGNORE_INDEX]
        counts += np.bincount(kept, minlength=NUM_PHASES)[:NUM_PHASES]
    total = max(int(counts.sum()), 1)
    return {name: float(counts[i] / total) for i, name in enumerate(PHASE_NAMES)}


def class_weights(distribution: dict[str, float], cap: float = 20.0) -> np.ndarray:
    """Inverse-frequency weights, capped.

    Uncapped inverse frequency on a set where FALLING is under 2% of frames
    produces weights in the hundreds, which destabilises training long before it
    improves anything - the same failure the binary loss caps `pos_weight` for.
    """
    weights = np.ones(NUM_PHASES, dtype=np.float32)
    for i, name in enumerate(PHASE_NAMES):
        share = distribution.get(name, 0.0)
        weights[i] = min(1.0 / share, cap) if share > 0 else 1.0
    return weights / weights.mean()
