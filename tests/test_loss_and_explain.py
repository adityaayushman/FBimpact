"""The pre-impact objective (Section 14) and the faithfulness machinery (Stage F).

The loss tests assert the property the objective exists for - that an earlier
correct warning is worth more - and that `lam = 0` is exactly plain BCE, since
the `- pre-impact loss` ablation depends on that equivalence being real rather
than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.labels import IGNORE_INDEX
from data.skeleton import NUM_JOINTS
from explain.faithfulness import apply_baseline, faithfulness_curves
from explain.relevance import joint_relevance
from losses.preimpact import PreImpactLoss, PreImpactLossConfig, time_weights
from models.stgcn import STGCN, STGCNConfig


@pytest.fixture
def model():
    torch.manual_seed(0)
    return STGCN(STGCNConfig(in_channels=4, blocks=((32, 1), (32, 1)), kernel_size=5)).eval()


def test_time_weights_favour_earlier_frames():
    tti = torch.tensor([[0.0, 0.2, 0.4, 0.6]])
    weights = time_weights(tti, lam=1.5, w_pre_seconds=0.67)
    assert torch.all(weights[0, 1:] > weights[0, :-1]), (
        "an earlier correct warning must carry more weight - that is the objective"
    )


def test_time_weights_average_to_one_over_the_window():
    """`lam` must not double as a learning-rate multiplier."""
    tti = torch.linspace(0.0, 0.67, 500).unsqueeze(0)
    for lam in (0.5, 1.5, 3.0):
        weights = time_weights(tti, lam=lam, w_pre_seconds=0.67)
        assert weights.mean().item() == pytest.approx(1.0, abs=0.02)


def test_time_weights_are_neutral_outside_the_window():
    tti = torch.tensor([[float("inf"), -0.5, 5.0]])   # ADL, post-impact, far before
    weights = time_weights(tti, lam=2.0, w_pre_seconds=0.67)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_lambda_zero_is_plain_bce():
    torch.manual_seed(0)
    logits = torch.randn(4, 30)
    labels = torch.randint(0, 2, (4, 30))
    tti = torch.rand(4, 30) * 0.6

    loss, _ = PreImpactLoss(PreImpactLossConfig(lam=0.0))(logits, labels, tti)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float()
    )
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5), (
        "the '- pre-impact loss' ablation must be exactly plain BCE, or it "
        "ablates two things at once"
    )


def test_ignored_frames_do_not_contribute():
    logits = torch.zeros(1, 10)
    labels = torch.zeros(1, 10, dtype=torch.long)
    tti = torch.full((1, 10), float("inf"))
    criterion = PreImpactLoss(PreImpactLossConfig(lam=0.0))

    baseline, _ = criterion(logits, labels, tti)

    poisoned = labels.clone()
    poisoned[0, 5:] = IGNORE_INDEX
    wild = logits.clone()
    wild[0, 5:] = 50.0                       # confidently wrong, but masked out
    masked, stats = criterion(wild, poisoned, tti)

    assert masked.item() == pytest.approx(baseline.item(), rel=1e-5)
    assert stats["valid_fraction"] == pytest.approx(0.5)


def test_pos_weight_from_prior_is_capped():
    criterion = PreImpactLoss(PreImpactLossConfig(max_pos_weight=20.0))
    assert criterion.set_pos_weight_from_prior(0.5) == pytest.approx(1.0)
    assert criterion.set_pos_weight_from_prior(0.0001) == 20.0, (
        "an uncapped weight from a rare-positive split destabilises training"
    )


def test_relevance_is_a_normalised_distribution(model):
    window = torch.randn(4, 30, NUM_JOINTS)
    for method in ("attention", "gradient_input", "occlusion"):
        relevance = joint_relevance(model, window, -1, torch.device("cpu"), method=method)
        assert relevance.scores.shape == (NUM_JOINTS,)
        assert relevance.scores.min() >= 0.0
        assert relevance.scores.sum() == pytest.approx(1.0)
        assert len(relevance.top_k(3)) == 3
        assert relevance.phrase(3)


def test_apply_baseline_only_touches_the_named_joints():
    window = torch.randn(1, 4, 30, NUM_JOINTS, 1)
    out = apply_baseline(window, [3, 7], "zero")
    assert torch.all(out[:, :, :, 3, :] == 0)
    assert torch.all(out[:, :, :, 7, :] == 0)
    untouched = [j for j in range(NUM_JOINTS) if j not in (3, 7)]
    assert torch.allclose(out[:, :, :, untouched, :], window[:, :, :, untouched, :])


def test_neighbour_baseline_stays_finite_and_in_range():
    window = torch.randn(1, 4, 30, NUM_JOINTS, 1)
    out = apply_baseline(window, [13, 14], "neighbour")
    assert torch.isfinite(out).all()
    assert out.abs().max() <= window.abs().max() + 1e-5, (
        "a neighbour average must interpolate, never extrapolate"
    )


def test_faithfulness_curves_have_the_right_shape_and_endpoints(model):
    window = torch.randn(4, 30, NUM_JOINTS)
    relevance = np.linspace(1.0, 0.0, NUM_JOINTS)
    curves = faithfulness_curves(
        model, window, -1, relevance, torch.device("cpu"), num_random=2, seed=0
    )

    assert curves.deletion.shape == (NUM_JOINTS + 1,)
    assert curves.insertion.shape == (NUM_JOINTS + 1,)
    # Deleting nothing and inserting everything are the same input.
    assert curves.deletion[0] == pytest.approx(curves.insertion[-1], abs=1e-6)
    # Deleting everything and inserting nothing are the same input.
    assert curves.deletion[-1] == pytest.approx(curves.insertion[0], abs=1e-6)
    assert 0.0 <= curves.deletion_auc <= 1.0
    assert 0.0 <= curves.insertion_auc <= 1.0


def test_faithfulness_detects_a_perfect_ranking():
    """A model that reads exactly one joint must give that joint a large gap."""

    class OneJointModel(torch.nn.Module):
        """Score depends only on joint 5, so the true ranking is known."""

        def forward(self, x, return_aux=False):
            logits = x[:, 0, :, 5, 0] * 10.0
            if return_aux:
                return logits, {}
            return logits

    model = OneJointModel().eval()
    window = torch.ones(1, 4, 30, NUM_JOINTS, 1)

    good = np.zeros(NUM_JOINTS)
    good[5] = 1.0                                     # the correct ranking
    curves = faithfulness_curves(
        model, window, -1, good, torch.device("cpu"), num_random=8, seed=0
    )
    assert curves.deletion_gap > 0.1, "a known-correct ranking scored no better than random"
    assert curves.insertion_gap > 0.1

    bad = np.ones(NUM_JOINTS)
    bad[5] = 0.0                                      # the exactly wrong ranking
    wrong = faithfulness_curves(
        model, window, -1, bad, torch.device("cpu"), num_random=8, seed=0
    )
    assert wrong.deletion_gap < curves.deletion_gap
