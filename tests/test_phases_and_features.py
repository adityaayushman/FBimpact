"""Phase labelling, richer features, and the multi-task objective.

Two of these tests exist because the bug they describe was actually shipped and
caught late: `test_pad_clip_pads_phases_too` (a misalignment that would have
been silent, and only in `ClipDataset`) and
`test_binary_builder_tolerates_multitask_keys` (which made every phases-off
config fail at start-up). Both are cheap to assert and expensive to notice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.features import (
    FEATURE_SETS,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    bone_vectors,
    build_features,
    channels_for,
    trunk_angle,
)
from data.labels import IGNORE_INDEX, LabelConfig, frame_labels
from data.phases import (
    FALLING,
    GROUNDED,
    IMMINENT,
    NORMAL,
    NUM_PHASES,
    PhaseConfig,
    class_weights,
    phase_distribution,
    phase_labels,
    to_binary,
)
from data.skeleton import BONES, NUM_JOINTS
from data.windows import pad_clip


# -- phases -----------------------------------------------------------------

def test_phase_boundaries_with_annotated_onset():
    phases = phase_labels(100, impact_frame=60, onset_frame=45,
                          config=PhaseConfig(w_pre=10))
    assert phases[34] == NORMAL
    assert phases[35] == IMMINENT      # onset - w_pre
    assert phases[44] == IMMINENT
    assert phases[45] == FALLING       # onset
    assert phases[60] == FALLING       # impact itself is still pre-contact
    assert phases[61] == GROUNDED


def test_missing_onset_falls_back_to_a_single_imminent_block():
    """A dataset that annotates only `t*` must not have a falling phase invented."""
    phases = phase_labels(100, impact_frame=60, onset_frame=None,
                          config=PhaseConfig(w_pre=20))
    assert FALLING not in set(phases.tolist())
    assert (phases[40:61] == IMMINENT).all()
    assert phases[39] == NORMAL


def test_adl_clip_is_entirely_normal():
    phases = phase_labels(80, impact_frame=None)
    assert (phases == NORMAL).all()


def test_to_binary_reproduces_the_binary_labeller_exactly():
    """The compatibility guarantee: phase training must not change the target."""
    for w_pre in (10, 20, 30):
        phases = phase_labels(120, impact_frame=70, onset_frame=None,
                              config=PhaseConfig(w_pre=w_pre))
        direct = frame_labels(120, impact_frame=70, config=LabelConfig(w_pre=w_pre))
        assert np.array_equal(to_binary(phases), direct)


def test_to_binary_treats_falling_as_positive():
    phases = phase_labels(100, impact_frame=60, onset_frame=40,
                          config=PhaseConfig(w_pre=10))
    binary = to_binary(phases)
    assert binary[45] == 1                      # falling
    assert binary[35] == 1                      # imminent
    assert binary[61] == IGNORE_INDEX           # grounded is masked, as before


def test_include_grounded_false_masks_post_impact():
    phases = phase_labels(100, 60, 45, PhaseConfig(w_pre=10, include_grounded=False))
    assert (phases[61:] == IGNORE_INDEX).all()
    assert GROUNDED not in set(phases.tolist())


def test_class_weights_are_capped_and_normalised():
    lopsided = {"normal": 0.97, "imminent": 0.001, "falling": 0.019, "grounded": 0.01}
    weights = class_weights(lopsided, cap=20.0)
    assert weights.shape == (NUM_PHASES,)
    assert np.all(np.isfinite(weights))
    # Uncapped inverse frequency would give 1000 for imminent.
    assert weights.max() / weights.min() <= 20.0 / 1.0 * 1.01
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)


def test_phase_distribution_ignores_masked_frames():
    a = phase_labels(60, 40, 30, PhaseConfig(w_pre=10))
    b = np.full(60, IGNORE_INDEX, dtype=np.int64)
    distribution = phase_distribution([a, b])
    assert sum(distribution.values()) == pytest.approx(1.0)


# -- features ---------------------------------------------------------------

@pytest.fixture
def moving_skeleton():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(NUM_JOINTS, 2)).astype(np.float32)
    drift = np.linspace(0, 1, 40, dtype=np.float32)[:, None, None]
    return base[None] + drift * 0.3


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_sets_have_the_declared_channel_count(feature_set, moving_skeleton):
    features = build_features(moving_skeleton, fps=30.0, feature_set=feature_set)
    assert features.shape == (channels_for(feature_set), 40, NUM_JOINTS)
    assert np.isfinite(features).all()


def test_derivatives_are_causal(moving_skeleton):
    """Perturbing a later frame must not change an earlier feature.

    The same property the model's convolutions have to satisfy - a centred
    difference here would leak the future into every channel.
    """
    clean = build_features(moving_skeleton, 30.0, "full")
    poked = moving_skeleton.copy()
    poked[25:] += 5.0
    dirty = build_features(poked, 30.0, "full")
    assert np.allclose(clean[:, :25], dirty[:, :25], atol=1e-5)
    assert not np.allclose(clean[:, 25:], dirty[:, 25:], atol=1e-3)


def test_derivatives_are_clipped(moving_skeleton):
    """A joint that drops out and snaps back must not set the channel's scale."""
    spiked = moving_skeleton.copy()
    spiked[20, 5] += 500.0            # one frame of impossible displacement
    features = build_features(spiked, 30.0, "xyva")
    velocity, acceleration = features[2:4], features[4:6]
    assert np.abs(velocity).max() <= MAX_VELOCITY + 1e-3
    assert np.abs(acceleration).max() <= MAX_ACCELERATION + 1e-3


def test_bone_vectors_are_translation_invariant(moving_skeleton):
    shifted = moving_skeleton + np.array([7.0, -3.0], dtype=np.float32)
    assert np.allclose(bone_vectors(moving_skeleton), bone_vectors(shifted), atol=1e-5)


def test_bone_vectors_follow_the_declared_parent_tree():
    """Every joint uses its one declared parent, not whichever bone came last.

    Joint 12 is a child of both 6 and 11 in `BONES`, so deriving bones from that
    list alone makes the result depend on declaration order.
    """
    from data.features import JOINT_PARENTS

    xy = np.zeros((1, NUM_JOINTS, 2), dtype=np.float32)
    xy[0, :, 0] = np.arange(NUM_JOINTS)
    bones = bone_vectors(xy)

    assert set(JOINT_PARENTS) == set(range(NUM_JOINTS)), "every joint needs a parent entry"
    for child, parent in JOINT_PARENTS.items():
        anchor = (
            sum(parent) / len(parent) if isinstance(parent, tuple) else float(parent)
        )
        assert bones[0, child, 0] == pytest.approx(child - anchor)


def test_bone_tree_is_symmetric_under_the_flip_augmentation():
    """Mirroring must give a mirrored tree, not a different one.

    Hanging the head off one shoulder, or rooting at one hip, breaks this: the
    flipped skeleton becomes a different tree rather than a reflected one, and
    the bone features stop being mirror images of the originals.
    """
    from data.features import JOINT_PARENTS
    from data.skeleton import flip_permutation

    perm = flip_permutation()

    def mirrored(parent):
        if isinstance(parent, tuple):
            return frozenset(int(perm[p]) for p in parent)
        return int(perm[parent])

    def canonical(parent):
        return frozenset(parent) if isinstance(parent, tuple) else parent

    for child, parent in JOINT_PARENTS.items():
        assert canonical(JOINT_PARENTS[int(perm[child])]) == mirrored(parent)


def test_trunk_angle_is_zero_upright_and_larger_when_leaning():
    from data.synthetic import _pose

    upright = _pose(np.zeros(2, dtype=np.float32), 0.0, 100.0, 0.1, 0.1, 0.0, 0.0)
    leaning = _pose(np.zeros(2, dtype=np.float32), 1.2, 100.0, 0.1, 0.1, 0.0, 0.0)
    a = trunk_angle(upright[None])[0]
    b = trunk_angle(leaning[None])[0]
    assert a == pytest.approx(0.0, abs=1e-3)
    assert b > a + 1.0


def test_xyv_matches_the_original_representation(moving_skeleton):
    """`xyv` must be exactly what every earlier result was trained on."""
    from data.normalize import add_velocity

    assert np.allclose(
        build_features(moving_skeleton, 30.0, "xyv"),
        add_velocity(moving_skeleton, fps=30.0),
        atol=1e-6,
    )


# -- padding ----------------------------------------------------------------

def test_pad_clip_pads_phases_too():
    """Regression: an unpadded phase array misaligns every label by `pad`."""
    features = np.zeros((4, 10, NUM_JOINTS), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)
    tti = np.zeros(10, dtype=np.float32)
    phases = np.full(10, FALLING, dtype=np.int64)

    padded_f, padded_l, padded_t, padded_p = pad_clip(features, labels, tti, 25, phases)
    assert padded_f.shape[1] == padded_l.shape[0] == padded_t.shape[0] == padded_p.shape[0] == 25
    assert (padded_p[:15] == IGNORE_INDEX).all()
    assert (padded_p[15:] == FALLING).all()


def test_pad_clip_without_phases_keeps_the_three_tuple():
    features = np.zeros((4, 10, NUM_JOINTS), dtype=np.float32)
    result = pad_clip(features, np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.float32), 25)
    assert len(result) == 3


# -- multi-task objective ---------------------------------------------------

def test_phase_weight_zero_recovers_the_binary_loss_exactly():
    """The ablation guarantee: the new component must switch off cleanly."""
    from losses.multitask import MultiTaskLoss, MultiTaskLossConfig
    from losses.preimpact import PreImpactLoss, PreImpactLossConfig

    torch.manual_seed(0)
    logits = torch.randn(3, 20)
    labels = torch.randint(0, 2, (3, 20))
    tti = torch.rand(3, 20)
    phase_logits = torch.randn(3, NUM_PHASES, 20)
    phases = torch.randint(0, NUM_PHASES, (3, 20))

    binary_only = PreImpactLoss(PreImpactLossConfig(lam=1.5))
    combined = MultiTaskLoss(MultiTaskLossConfig(
        binary=PreImpactLossConfig(lam=1.5), phase_weight=0.0, consistency_weight=0.0
    ))
    expected, _ = binary_only(logits, labels, tti)
    actual, _ = combined(logits, labels, tti, phase_logits, phases)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_phase_term_increases_the_loss_when_the_phase_head_is_wrong():
    from losses.multitask import MultiTaskLoss, MultiTaskLossConfig
    from losses.preimpact import PreImpactLossConfig

    torch.manual_seed(0)
    logits = torch.zeros(2, 12)
    labels = torch.zeros(2, 12, dtype=torch.long)
    tti = torch.full((2, 12), float("inf"))
    phases = torch.full((2, 12), FALLING, dtype=torch.long)

    confident_wrong = torch.zeros(2, NUM_PHASES, 12)
    confident_wrong[:, NORMAL] = 10.0
    confident_right = torch.zeros(2, NUM_PHASES, 12)
    confident_right[:, FALLING] = 10.0

    loss = MultiTaskLoss(MultiTaskLossConfig(
        binary=PreImpactLossConfig(lam=0.0), phase_weight=1.0, consistency_weight=0.0
    ))
    wrong, _ = loss(logits, labels, tti, confident_wrong, phases)
    right, stats = loss(logits, labels, tti, confident_right, phases)
    assert wrong.item() > right.item()
    assert stats["phase_accuracy"] == pytest.approx(1.0)


def test_masked_phases_contribute_nothing():
    from losses.multitask import MultiTaskLoss, MultiTaskLossConfig
    from losses.preimpact import PreImpactLossConfig

    torch.manual_seed(0)
    logits = torch.randn(2, 10)
    labels = torch.zeros(2, 10, dtype=torch.long)
    tti = torch.full((2, 10), float("inf"))
    phase_logits = torch.randn(2, NUM_PHASES, 10) * 10.0
    masked = torch.full((2, 10), IGNORE_INDEX, dtype=torch.long)

    loss = MultiTaskLoss(MultiTaskLossConfig(
        binary=PreImpactLossConfig(lam=0.0), phase_weight=1.0, consistency_weight=1.0
    ))
    combined, _ = loss(logits, labels, tti, phase_logits, masked)
    binary, _ = loss.binary_loss(logits, labels, tti)
    assert combined.item() == pytest.approx(binary.item(), rel=1e-6)


def test_binary_builder_tolerates_multitask_keys():
    """Regression: these keys live in every config's shared `loss` block."""
    from losses.preimpact import build_loss

    built = build_loss({"lam": 0.0, "phase_weight": 0.5, "consistency_weight": 0.1})
    assert built.config.lam == 0.0


def test_binary_builder_still_rejects_genuine_typos():
    from losses.preimpact import build_loss

    with pytest.raises(ValueError, match="unknown loss options"):
        build_loss({"lam": 0.0, "lambdaa": 1.0})


def test_phase_head_preserves_causality():
    """The property the whole lead-time claim rests on, re-checked with two heads."""
    from models.stgcn import STGCN, STGCNConfig

    torch.manual_seed(0)
    model = STGCN(STGCNConfig(
        in_channels=14, blocks=((32, 1), (32, 1)), kernel_size=5, phase_head=True
    )).eval()
    x = torch.randn(1, 14, 30, NUM_JOINTS, 1)

    with torch.no_grad():
        before, aux_before = model(x, return_aux=True)
        poked = x.clone()
        poked[:, :, 20:] += 5.0
        after, aux_after = model(poked, return_aux=True)

    assert torch.allclose(before[:, :20], after[:, :20], atol=1e-5)
    assert torch.allclose(
        aux_before["phase_logits"][:, :, :20], aux_after["phase_logits"][:, :, :20], atol=1e-5
    )
