"""Guards the torch-free re-implementation of phase counting against the real one.

`scripts/export_datasets.py` cannot import `data.phases` (that pulls in torch
through the package `__init__`), so it duplicates the arithmetic. A duplicate that
is not checked eventually disagrees with its original, and the Data pane would then
describe a labelling the models were never trained on. This test imports the real
labeller - so it needs torch and belongs with the tests run in a training pause.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_datasets as ed  # noqa: E402

phases = pytest.importorskip("data.phases")


CASES = [
    (300, None, None, 17),       # normal-activity clip
    (300, 200, None, 17),        # fall, no annotated onset
    (300, 200, 170, 17),         # fall with onset
    (300, 200, 190, 17),         # onset close to impact
    (300, 12, 5, 17),            # impact before a full window exists
    (120, 119, 100, 17),         # impact on the last frame (no grounded frames)
    (300, 200, 2, 17),           # onset very early: pre-onset window clipped at 0
]


@pytest.mark.parametrize("frames,impact,onset,w_pre", CASES)
def test_phase_counts_match_the_real_labeller(frames, impact, onset, w_pre):
    real = phases.phase_labels(frames, impact, onset, phases.PhaseConfig(w_pre=w_pre))
    expected = np.bincount(real[real >= 0], minlength=4)
    assert ed.phase_counts(frames, impact, onset, w_pre).tolist() == expected.tolist()
