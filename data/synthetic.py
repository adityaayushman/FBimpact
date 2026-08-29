"""A procedural skeleton generator, so the pipeline runs before any download.

This is a **smoke-test fixture, not a dataset**. It produces plausible COCO-17
trajectories for a handful of activities and falls with a known impact frame,
which is enough to exercise every stage end to end - windowing, labelling,
training, decision logic, faithfulness - and to write tests that assert on real
numbers. No result produced on it belongs in the paper; UP-Fall is the benchmark
(Section 13).

The activity list is deliberately weighted towards *hard negatives*: sitting
down, bending to pick something up and lying down are all controlled descents.
A model that fires on any downward motion will score well on a synthetic set of
walking versus falling and then fall apart on real ADL data, so the fixture is
built to punish that shortcut from the start.
"""

from __future__ import annotations

import numpy as np

from .clips import ClipRecord
from .skeleton import JOINT_INDEX, NUM_JOINTS

FALL_ACTIVITIES = ("fall_forward", "fall_backward", "fall_sideways")
ADL_ACTIVITIES = ("walk", "sit_down", "stand_up", "pick_up", "lie_down")

# Limb lengths as a fraction of torso length.
_PROPORTIONS = {
    "hip_half_width": 0.18,
    "shoulder_half_width": 0.22,
    "neck": 0.16,
    "head": 0.22,
    "upper_arm": 0.32,
    "forearm": 0.30,
    "thigh": 0.50,
    "shank": 0.48,
}


def _rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-vector by `angle` radians (image coordinates, y downward)."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([vec[0] * cos - vec[1] * sin, vec[0] * sin + vec[1] * cos], dtype=np.float32)


def _pose(
    root: np.ndarray,
    trunk_angle: float,
    torso: float,
    hip_flex: float,
    knee_flex: float,
    arm_swing: float,
    arm_raise: float,
) -> np.ndarray:
    """Build one `[V, 2]` skeleton from joint angles.

    `trunk_angle` is measured from vertical: 0 is upright, +pi/2 is lying with
    the head in the +x direction. `hip_flex` and `knee_flex` are the thigh and
    shank deviations from straight-down, which is how sitting, crouching and
    knee buckling are all expressed.
    """
    joints = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    p = _PROPORTIONS
    up = np.array([0.0, -1.0], dtype=np.float32)

    trunk_dir = _rotate(up, trunk_angle)
    # Perpendicular to the trunk, used to offset the paired joints sideways.
    side = np.array([trunk_dir[1], -trunk_dir[0]], dtype=np.float32)

    hip_c = root
    joints[JOINT_INDEX["left_hip"]] = hip_c + side * (p["hip_half_width"] * torso)
    joints[JOINT_INDEX["right_hip"]] = hip_c - side * (p["hip_half_width"] * torso)

    shoulder_c = hip_c + trunk_dir * torso
    joints[JOINT_INDEX["left_shoulder"]] = shoulder_c + side * (p["shoulder_half_width"] * torso)
    joints[JOINT_INDEX["right_shoulder"]] = shoulder_c - side * (p["shoulder_half_width"] * torso)

    neck = shoulder_c + trunk_dir * (p["neck"] * torso)
    nose = neck + trunk_dir * (p["head"] * torso)
    joints[JOINT_INDEX["nose"]] = nose
    joints[JOINT_INDEX["left_eye"]] = nose + side * (0.04 * torso) - trunk_dir * (0.02 * torso)
    joints[JOINT_INDEX["right_eye"]] = nose - side * (0.04 * torso) - trunk_dir * (0.02 * torso)
    joints[JOINT_INDEX["left_ear"]] = neck + side * (0.08 * torso) + trunk_dir * (0.12 * torso)
    joints[JOINT_INDEX["right_ear"]] = neck - side * (0.08 * torso) + trunk_dir * (0.12 * torso)

    for sign, tag in ((1.0, "left"), (-1.0, "right")):
        shoulder = joints[JOINT_INDEX[f"{tag}_shoulder"]]
        upper = _rotate(-trunk_dir, sign * (arm_swing + arm_raise))
        elbow = shoulder + upper * (p["upper_arm"] * torso)
        forearm = _rotate(upper, sign * 0.35)
        joints[JOINT_INDEX[f"{tag}_elbow"]] = elbow
        joints[JOINT_INDEX[f"{tag}_wrist"]] = elbow + forearm * (p["forearm"] * torso)

    for sign, tag in ((1.0, "left"), (-1.0, "right")):
        hip = joints[JOINT_INDEX[f"{tag}_hip"]]
        thigh_dir = _rotate(np.array([0.0, 1.0], dtype=np.float32), hip_flex * sign)
        knee = hip + thigh_dir * (p["thigh"] * torso)
        shank_dir = _rotate(thigh_dir, -knee_flex)
        joints[JOINT_INDEX[f"{tag}_knee"]] = knee
        joints[JOINT_INDEX[f"{tag}_ankle"]] = knee + shank_dir * (p["shank"] * torso)

    return joints


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Ease-in/ease-out on `[0, 1]`, for controlled (non-ballistic) motions."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _adl_trajectory(activity: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Parameter trajectories for a normal activity."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    params = {
        "trunk": zeros + rng.normal(0.0, 0.02),
        "drop": zeros.copy(),
        "hip_flex": zeros.copy(),
        "knee_flex": zeros + 0.05,
        "arm_swing": zeros.copy(),
        "arm_raise": zeros.copy(),
        "drift": zeros.copy(),
    }

    if activity == "walk":
        stride = rng.uniform(1.6, 2.6)
        phase = 2.0 * np.pi * stride * t + rng.uniform(0.0, 2.0 * np.pi)
        params["drift"] = t * rng.uniform(-1.4, 1.4)
        params["drop"] = 0.02 * np.sin(2.0 * phase)
        params["hip_flex"] = 0.32 * np.sin(phase)
        params["knee_flex"] = 0.18 + 0.16 * np.clip(np.sin(phase + 0.8), 0.0, None)
        params["arm_swing"] = 0.35 * np.sin(phase + np.pi)
        params["trunk"] += 0.04 * np.sin(phase)

    elif activity in ("sit_down", "stand_up"):
        ramp = _smoothstep((t - 0.25) / 0.5)
        if activity == "stand_up":
            ramp = 1.0 - ramp
        params["drop"] = 0.42 * ramp
        params["knee_flex"] = 0.05 + 1.25 * ramp
        params["hip_flex"] = 0.85 * ramp
        params["trunk"] += 0.30 * ramp          # a real lean, just a controlled one
        params["arm_raise"] = 0.20 * ramp

    elif activity == "pick_up":
        bend = np.sin(np.pi * _smoothstep((t - 0.15) / 0.7))
        params["trunk"] += 1.05 * bend           # far more lean than a fall's onset
        params["drop"] = 0.20 * bend
        params["knee_flex"] = 0.05 + 0.55 * bend
        params["arm_raise"] = 0.9 * bend

    elif activity == "lie_down":
        ramp = _smoothstep((t - 0.2) / 0.6)
        params["trunk"] += (np.pi / 2.0 - 0.12) * ramp
        params["drop"] = 0.78 * ramp
        params["knee_flex"] = 0.05 + 0.40 * ramp
        params["hip_flex"] = 0.55 * ramp

    else:
        raise ValueError(f"unknown ADL activity {activity!r}")

    return params


def _fall_trajectory(
    activity: str, n: int, impact: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Parameter trajectories for a fall with ground contact at `impact`.

    The pre-impact phase is *ballistic*: lean and drop follow a squared ramp so
    their velocity keeps increasing right up to contact, unlike the eased ramps
    of sitting and lying down. That difference in the second derivative is the
    signal the model is meant to find, and it is the reason velocity is a feature.
    """
    t = np.arange(n, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)

    onset = max(1, impact - int(rng.integers(18, 34)))
    ramp = np.clip((t - onset) / max(impact - onset, 1), 0.0, None)
    pre = np.minimum(ramp, 1.0) ** 2.0                      # accelerating collapse
    settle = _smoothstep((t - impact) / 8.0) * (t > impact)  # bounce/settle after contact

    direction = {"fall_forward": 1.0, "fall_backward": -1.0, "fall_sideways": 0.7}[activity]
    final_lean = direction * (np.pi / 2.0 - rng.uniform(0.05, 0.25))

    params = {
        "trunk": final_lean * pre + rng.normal(0.0, 0.02),
        "drop": 0.86 * pre + 0.05 * settle,
        "hip_flex": 0.35 * pre,
        "knee_flex": 0.05 + 0.95 * pre * (1.0 - 0.4 * pre),   # buckle, then extend
        "arm_swing": zeros.copy(),
        "arm_raise": 1.15 * np.clip((t - onset) / max(impact - onset, 1), 0.0, 1.0),
        "drift": 0.35 * pre * direction,
    }
    # Before the onset the person is simply standing, with a little sway.
    sway = 0.03 * np.sin(2.0 * np.pi * t / max(n, 1) * rng.uniform(1.0, 2.0))
    params["trunk"] = params["trunk"] + sway * (t < onset)
    return params


def _render(
    params: dict[str, np.ndarray],
    torso_px: float,
    origin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn parameter trajectories into `[T, V, 3]` pixel keypoints."""
    n = len(params["trunk"])
    keypoints = np.zeros((n, NUM_JOINTS, 3), dtype=np.float32)

    for i in range(n):
        root = origin + np.array(
            [params["drift"][i] * torso_px, params["drop"][i] * torso_px], dtype=np.float32
        )
        keypoints[i, :, :2] = _pose(
            root=root,
            trunk_angle=float(params["trunk"][i]),
            torso=torso_px,
            hip_flex=float(params["hip_flex"][i]),
            knee_flex=float(params["knee_flex"][i]),
            arm_swing=float(params["arm_swing"][i]),
            arm_raise=float(params["arm_raise"][i]),
        )

    # Pose-estimator jitter, plus a slow wobble that a per-frame model could
    # otherwise mistake for real motion.
    keypoints[:, :, :2] += rng.normal(0.0, 0.006 * torso_px, size=(n, NUM_JOINTS, 2))
    keypoints[:, :, :2] += rng.normal(0.0, 0.004 * torso_px, size=(1, NUM_JOINTS, 2))

    conf = rng.uniform(0.72, 0.98, size=(n, NUM_JOINTS)).astype(np.float32)
    # Section 18: pose estimators lose the lower body when a person is down.
    low = np.flatnonzero(params["drop"] > 0.5)
    if low.size:
        conf[np.ix_(low, [13, 14, 15, 16])] *= 0.55
    dropout = rng.random((n, NUM_JOINTS)) < 0.03
    conf[dropout] = rng.uniform(0.0, 0.25, size=int(dropout.sum())).astype(np.float32)
    keypoints[:, :, 2] = conf
    return keypoints


def make_clip(
    clip_id: str,
    subject: str,
    activity: str,
    rng: np.random.Generator,
    fps: float = 30.0,
    num_frames: int | None = None,
) -> ClipRecord:
    """Generate one synthetic clip."""
    n = num_frames or int(rng.integers(90, 150))
    torso_px = float(rng.uniform(95.0, 135.0))
    origin = np.array([rng.uniform(260.0, 380.0), rng.uniform(230.0, 290.0)], dtype=np.float32)

    if activity in FALL_ACTIVITIES:
        impact = int(rng.integers(int(n * 0.55), int(n * 0.80)))
        params = _fall_trajectory(activity, n, impact, rng)
        label, impact_frame = "fall", impact
    else:
        params = _adl_trajectory(activity, n, rng)
        label, impact_frame = "adl", None

    return ClipRecord(
        clip_id=clip_id,
        subject=subject,
        keypoints=_render(params, torso_px, origin, rng),
        fps=fps,
        label=label,
        impact_frame=impact_frame,
        activity=activity,
        view=f"cam{rng.integers(1, 3)}",
        source="synthetic",
        meta={"torso_px": torso_px},
    )


def make_dataset(
    num_subjects: int = 10,
    falls_per_subject: int = 6,
    adls_per_subject: int = 10,
    fps: float = 30.0,
    seed: int = 0,
) -> list[ClipRecord]:
    """Generate a full synthetic cache.

    Each subject gets its own body proportions and motion idiosyncrasies through
    a per-subject seed, so a subject-independent split is a genuine test of
    generalisation rather than a formality.
    """
    clips: list[ClipRecord] = []
    for s in range(num_subjects):
        subject = f"S{s:02d}"
        rng = np.random.default_rng((seed, s))
        for i in range(falls_per_subject):
            activity = FALL_ACTIVITIES[i % len(FALL_ACTIVITIES)]
            clips.append(make_clip(f"{subject}_fall{i:02d}", subject, activity, rng, fps))
        for i in range(adls_per_subject):
            activity = ADL_ACTIVITIES[i % len(ADL_ACTIVITIES)]
            clips.append(make_clip(f"{subject}_adl{i:02d}", subject, activity, rng, fps))
    return clips
