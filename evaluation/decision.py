"""Stage G - turning a noisy score stream into a warning.

Section 10: warn when `p_t >= tau` for `k` consecutive frames. Two details that
the rule leaves open decide whether the reported lead time is honest.

**Which of the `k` frames is `t_warn`.** It is the *last* one. The system cannot
know that a run of `k` frames has occurred until the `k`-th has arrived, so
timestamping the warning at the first frame of the run would credit the model
with `(k - 1) / fps` seconds it did not have - 67 ms at the default `k = 3` and
30 fps, which is a material fraction of the lead times being compared.

**How repeat triggers are counted.** A single sustained high-score episode is one
alarm to a caregiver, not one per frame, so a refractory period suppresses
re-triggering. Counting per frame would inflate the false-alarm rate by roughly
the frame rate and make the operating-point curve meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionConfig:
    """Trigger rule parameters."""

    threshold: float = 0.70
    persistence: int = 3
    """`k`: consecutive frames at or above the threshold."""

    refractory_frames: int = 30
    """Frames suppressed after a trigger before another can fire (~1 s)."""

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.persistence < 1:
            raise ValueError(f"persistence must be >= 1, got {self.persistence}")


def _run_ends(above: np.ndarray, k: int) -> np.ndarray:
    """Indices where a run of `k` consecutive True values completes.

    Vectorised, because the operating-point sweep calls this once per clip per
    `(tau, k)` pair - a few hundred thousand times per validation pass.
    """
    above = np.asarray(above, dtype=bool)
    if k == 1:
        return np.flatnonzero(above)
    index = np.arange(above.size)
    # Index of the most recent False at or before each position; -1 if none yet.
    last_false = np.maximum.accumulate(np.where(~above, index, -1))
    run_length = np.where(above, index - last_false, 0)
    return np.flatnonzero(run_length >= k)


def first_trigger(scores: np.ndarray, config: DecisionConfig) -> int | None:
    """Frame at which the first warning fires, or None if it never does."""
    ends = _run_ends(np.asarray(scores) >= config.threshold, config.persistence)
    return int(ends[0]) if ends.size else None


def all_triggers(scores: np.ndarray, config: DecisionConfig) -> list[int]:
    """Every warning frame, with the refractory period applied.

    Args:
        scores: `[T]` per-frame probabilities.
        config: trigger rule.

    Returns:
        Frame indices of the alarms a caregiver would actually receive.
    """
    ends = _run_ends(np.asarray(scores) >= config.threshold, config.persistence)
    triggers: list[int] = []
    blocked_until = -1
    for end in ends:
        if end < blocked_until:
            continue
        triggers.append(int(end))
        blocked_until = int(end) + config.refractory_frames
    return triggers


class OnlineTrigger:
    """Streaming version of the same rule, for `infer.py`.

    Keeps only a counter and a countdown, so the decision stage adds nothing to
    the memory footprint of a real-time deployment.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self.reset()

    def reset(self) -> None:
        self._run = 0
        self._cooldown = 0
        self.frame = -1

    def update(self, score: float) -> bool:
        """Feed one frame's score; returns True on the frame a warning fires."""
        self.frame += 1
        self._run = self._run + 1 if score >= self.config.threshold else 0
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if self._run >= self.config.persistence:
            self._cooldown = self.config.refractory_frames
            return True
        return False


def sweep_grid(
    thresholds: np.ndarray | None = None,
    persistences: tuple[int, ...] = (1, 2, 3, 5, 8),
    refractory_frames: int = 30,
) -> list[DecisionConfig]:
    """Operating points for the lead-time / false-alarm curve (Section 10)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 3)
    return [
        DecisionConfig(
            threshold=float(t), persistence=int(k), refractory_frames=refractory_frames
        )
        for k in persistences
        for t in thresholds
    ]
