"""Stages G-H: the trigger rule, lead time, and what counts as a false alarm.

These pin down the conventions that decide the headline number - where `t_warn`
sits inside a run of `k` frames, and how a warning fired before the imminent
window is scored. Both are choices, and a change to either moves every lead time
in the paper, so they are asserted rather than left to be rediscovered.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.clips import ClipRecord
from data.labels import LabelConfig, frame_labels, time_to_impact
from data.skeleton import NUM_JOINTS
from evaluation.decision import DecisionConfig, OnlineTrigger, all_triggers, first_trigger
from evaluation.metrics import ClipScores, evaluate, evaluate_clip


def make_clip(clip_id="c", num_frames=100, impact=60, fps=30.0, label="fall") -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        subject="S00",
        keypoints=np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32),
        fps=fps,
        label=label,
        impact_frame=impact if label == "fall" else None,
    )


def make_scores(clip: ClipRecord, scores: np.ndarray, w_pre=20) -> ClipScores:
    return ClipScores(
        clip=clip,
        scores=scores,
        labels=frame_labels(clip.num_frames, clip.impact_frame, LabelConfig(w_pre=w_pre)),
        tti=time_to_impact(clip.num_frames, clip.impact_frame, clip.fps),
    )


def test_trigger_fires_on_the_last_frame_of_the_run():
    """`t_warn` is the k-th frame: the system cannot know sooner that a run occurred."""
    scores = np.zeros(50)
    scores[10:20] = 0.9
    assert first_trigger(scores, DecisionConfig(threshold=0.7, persistence=3)) == 12
    assert first_trigger(scores, DecisionConfig(threshold=0.7, persistence=1)) == 10
    assert first_trigger(scores, DecisionConfig(threshold=0.7, persistence=5)) == 14


def test_persistence_rejects_a_short_spike():
    scores = np.zeros(50)
    scores[10:12] = 0.99                       # only two frames
    assert first_trigger(scores, DecisionConfig(threshold=0.7, persistence=3)) is None


def test_run_must_be_consecutive():
    scores = np.zeros(50)
    scores[[10, 12, 14, 16]] = 0.99            # alternating, never three in a row
    assert first_trigger(scores, DecisionConfig(threshold=0.7, persistence=3)) is None


def test_refractory_period_collapses_one_episode_into_one_alarm():
    scores = np.zeros(200)
    scores[10:100] = 0.99                      # one long sustained episode
    triggers = all_triggers(
        scores, DecisionConfig(threshold=0.7, persistence=3, refractory_frames=30)
    )
    assert len(triggers) == 3, (
        "without a refractory period a single episode would count as ~88 alarms "
        "and the false-alarm rate would be meaningless"
    )


def test_online_trigger_matches_the_offline_rule():
    rng = np.random.default_rng(0)
    scores = rng.random(300)
    config = DecisionConfig(threshold=0.6, persistence=3, refractory_frames=10)

    online = OnlineTrigger(config)
    streamed = [i for i, s in enumerate(scores) if online.update(float(s))]
    assert streamed == all_triggers(scores, config), (
        "the streaming and batch decision rules disagree, so the live system "
        "would not reproduce the reported numbers"
    )


def test_lead_time_is_measured_from_the_impact_frame():
    clip = make_clip(impact=60, fps=30.0)
    scores = np.zeros(100)
    scores[45:] = 0.99                          # trigger completes at frame 47
    outcome = evaluate_clip(
        make_scores(clip, scores), DecisionConfig(threshold=0.7, persistence=3), w_pre=20
    )
    assert outcome.warned
    assert outcome.trigger_frame == 47
    assert outcome.lead_time == pytest.approx((60 - 47) / 30.0)


def test_no_warning_before_impact_is_a_miss():
    clip = make_clip(impact=60)
    scores = np.zeros(100)
    scores[70:] = 0.99                          # only detects after the fall
    outcome = evaluate_clip(
        make_scores(clip, scores), DecisionConfig(threshold=0.7, persistence=3), w_pre=20
    )
    assert not outcome.warned
    assert outcome.lead_time is None


def test_warning_before_the_imminent_window_is_a_false_alarm_by_default():
    """Firing 1.5 s early is not a 1.5 s lead time - that frame is labelled normal."""
    clip = make_clip(impact=60, fps=30.0)
    scores = np.zeros(100)
    scores[5:15] = 0.99                         # completes at 7, well before t*-w_pre=40
    item = make_scores(clip, scores)
    config = DecisionConfig(threshold=0.7, persistence=3)

    default = evaluate_clip(item, config, w_pre=20, early_trigger="false_alarm")
    assert not default.warned
    assert default.early_trigger
    assert default.false_alarms == 1

    lenient = evaluate_clip(item, config, w_pre=20, early_trigger="hit")
    assert lenient.warned
    assert lenient.lead_time == pytest.approx((60 - 7) / 30.0)


def test_adl_trigger_counts_as_a_false_alarm():
    clip = make_clip(label="adl", num_frames=100)
    scores = np.zeros(100)
    scores[30:40] = 0.99
    outcome = evaluate_clip(
        make_scores(clip, scores), DecisionConfig(threshold=0.7, persistence=3), w_pre=20
    )
    assert not outcome.is_fall
    assert outcome.false_alarms == 1


def test_perfect_and_silent_models_bracket_the_metrics():
    falls = [make_clip(f"f{i}", impact=60) for i in range(4)]
    adls = [make_clip(f"a{i}", label="adl") for i in range(4)]
    config = DecisionConfig(threshold=0.7, persistence=3)

    perfect, silent = [], []
    for clip in falls:
        good = np.zeros(100)
        good[45:61] = 0.99
        perfect.append(make_scores(clip, good))
        silent.append(make_scores(clip, np.zeros(100)))
    for clip in adls:
        perfect.append(make_scores(clip, np.zeros(100)))
        silent.append(make_scores(clip, np.zeros(100)))

    good_report = evaluate(perfect, config, w_pre=20)
    assert good_report.recall == 1.0
    assert good_report.num_false_alarms == 0
    assert good_report.mean_lead_time == pytest.approx((60 - 47) / 30.0)

    bad_report = evaluate(silent, config, w_pre=20)
    assert bad_report.recall == 0.0
    assert np.isnan(bad_report.mean_lead_time), "lead time is undefined with no warnings"


def test_frame_auc_handles_ties_and_single_class():
    from evaluation.metrics import _auc_from_scores

    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert _auc_from_scores(scores, labels) == pytest.approx(0.5), (
        "a model that outputs a constant must score 0.5, not something decided "
        "by array order"
    )
    assert np.isnan(_auc_from_scores(np.array([0.1, 0.9]), np.array([0, 0])))
    assert _auc_from_scores(np.array([0.1, 0.9]), np.array([0, 1])) == pytest.approx(1.0)
