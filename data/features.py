"""Feature construction beyond raw coordinates and velocity.

The baseline representation is `(x, y, vx, vy)` - position and first derivative.
The ablation says velocity matters a great deal (removing it drops recall from
0.77 to 0.53 on UR Fall), which is a strong hint that the *derivatives* carry
the anticipation signal and that stopping at the first one is leaving something
on the table.

What is added, and why each earns its channels:

**Acceleration.** A fall is ballistic and a controlled descent is not: sitting
down decelerates into the chair, falling does not decelerate at all until
impact. That difference lives in the second derivative. A temporal convolution
can in principle recover it from positions, but handing it over directly is
cheaper than making the network learn a differencing kernel.

**Bone vectors.** Absolute joint positions confound posture with where the
person is standing. A bone vector - child minus parent - is translation
invariant by construction and describes limb *orientation*, which is what
"trunk lean" and "knee buckling" actually are. This is the second stream in
2s-AGCN and reliably helps in skeleton action recognition.

**Trunk angle from vertical, and its rate.** Gravity gives this task a
privileged direction, and no channel above encodes it explicitly. Trunk angle is
one number that captures the single most diagnostic thing about a fall, and its
rate distinguishes bending over (slow, reversing) from toppling (fast, monotonic).

**Centre-of-mass height and vertical velocity.** Scalar summaries of the whole
body that survive the loss of any individual joint - which matters when pose
coverage drops to 45% after impact, as measured on UR Fall.

Every feature is derived from the already-normalised skeleton, so nothing here
reintroduces translation or scale dependence, and all differences are **causal
backward differences** - a centred difference would leak one frame of the future
into every feature, which for anticipation means leaking the fall.
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    BONES,
    CENTRE_JOINTS,
    NUM_JOINTS,
    TORSO_BOTTOM_JOINTS,
    TORSO_TOP_JOINTS,
)

FEATURE_SETS = ("xy", "xyv", "xyva", "full")

# Coordinates are in torso lengths, so derivative units are torso lengths per
# second. A joint genuinely moves at a few torso lengths per second even during
# a fall; anything far beyond that is a pose-estimator artefact - a joint that
# dropped out and snapped back across the body produces one frame of enormous
# apparent speed. Left unclipped that single frame dominates the channel's batch
# statistics, and on this data acceleration reached 1210 against positions of 10.
# These caps are loose enough never to touch real motion and tight enough to
# stop a dropout from setting the scale.
MAX_VELOCITY = 30.0
MAX_ACCELERATION = 300.0


def channels_for(feature_set: str) -> int:
    """Channel count `C` produced by a named feature set."""
    return {
        "xy": 2,     # positions only - the `- velocity features` ablation
        "xyv": 4,    # positions + velocity - the current default
        "xyva": 6,   # + acceleration
        "full": 14,  # + bone, bone rate, trunk angle & rate, CoM height & rate
    }[feature_set]


def _backward_diff(x: np.ndarray, fps: float, clip: float | None = None) -> np.ndarray:
    """Causal first difference along time, scaled to units per second."""
    out = np.zeros_like(x)
    out[1:] = (x[1:] - x[:-1]) * float(fps)
    if len(x) > 1:
        # Frame 0 has no predecessor; copying frame 1 avoids a spurious spike at
        # the clip boundary that the model would otherwise learn to key on.
        out[0] = out[1]
    if clip is not None:
        np.clip(out, -clip, clip, out=out)
    return out


# Explicit parent for each joint, as a spanning tree over the skeleton graph.
#
# `BONES` is a graph, not a tree: joint 12 (right hip) is a child of both joint 6
# and joint 11. Deriving bones by iterating `BONES` leaves such joints holding
# whichever bone came last - a silent dependence on declaration order.
#
# A parent is either a joint index or a **tuple of joints whose midpoint is the
# parent**. The tuple form exists because COCO-17 has no pelvis or neck joint,
# and inventing an asymmetric substitute breaks the left/right flip
# augmentation: hanging the head off the left shoulder means a mirrored skeleton
# is a *different* tree, not a reflected one, and the bone features stop being
# mirror images. Both hips hang off the mid-hip and the nose off the
# mid-shoulder, so every parent relation is preserved under the flip.
JOINT_PARENTS: dict[int, int | tuple[int, ...]] = {
    11: (11, 12), 12: (11, 12),          # hips, from the mid-hip
    5: 11, 6: 12,                        # shoulders from the same-side hip
    7: 5, 9: 7, 8: 6, 10: 8,             # arms
    13: 11, 15: 13, 14: 12, 16: 14,      # legs
    0: (5, 6),                           # nose from the mid-shoulder
    1: 0, 2: 0, 3: 1, 4: 2,              # eyes and ears from the nose
}


def bone_vectors(xy: np.ndarray) -> np.ndarray:
    """Per-joint bone vector: the offset from each joint to its parent. `[T, V, 2]`.

    Translation invariant by construction, which is the point: absolute joint
    positions confound posture with where in the room the person is standing,
    whereas a bone describes limb orientation - what "trunk lean" and "knee
    buckling" actually are. Midpoint parents preserve that, because a midpoint of
    observed joints shifts with them.
    """
    bones = np.zeros_like(xy)
    for child, parent in JOINT_PARENTS.items():
        anchor = (
            xy[:, list(parent), :].mean(axis=1)
            if isinstance(parent, tuple)
            else xy[:, parent, :]
        )
        bones[:, child, :] = xy[:, child, :] - anchor
    return bones


def trunk_angle(xy: np.ndarray) -> np.ndarray:
    """Angle of the torso from vertical, in radians. `[T]`.

    Zero is upright and `pi/2` is horizontal, regardless of which way the person
    fell - the sign is dropped deliberately so that forward and backward falls
    present the same way to the model.
    """
    top = xy[:, list(TORSO_TOP_JOINTS), :].mean(axis=1)
    bottom = xy[:, list(TORSO_BOTTOM_JOINTS), :].mean(axis=1)
    trunk = top - bottom
    # Image coordinates have y increasing downward, so "up" is negative y.
    return np.arctan2(np.abs(trunk[:, 0]), np.maximum(-trunk[:, 1], 1e-6)).astype(np.float32)


def centre_of_mass(xy: np.ndarray) -> np.ndarray:
    """Mid-hip position as a whole-body summary. `[T, 2]`."""
    return xy[:, list(CENTRE_JOINTS), :].mean(axis=1)


def build_features(
    xy: np.ndarray,
    fps: float = 30.0,
    feature_set: str = "xyv",
) -> np.ndarray:
    """Assemble `[C, T, V]` features from normalised coordinates `[T, V, 2]`.

    Args:
        xy: normalised, gap-filled joint coordinates.
        fps: frame rate, so derivatives are per second and comparable across
            datasets recorded at 25 and 30 fps.
        feature_set: one of `FEATURE_SETS`.

    Returns:
        `[C, T, V]` float32.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")

    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1] != NUM_JOINTS or xy.shape[2] != 2:
        raise ValueError(f"expected [T, {NUM_JOINTS}, 2], got {xy.shape}")

    parts: list[np.ndarray] = [xy]                       # (x, y)

    if feature_set != "xy":
        velocity = _backward_diff(xy, fps, MAX_VELOCITY)
        parts.append(velocity)                            # (vx, vy)

    if feature_set in ("xyva", "full"):
        acceleration = _backward_diff(velocity, fps, MAX_ACCELERATION)
        parts.append(acceleration)                        # (ax, ay)

    if feature_set == "full":
        bones = bone_vectors(xy)
        parts.append(bones)                               # (bx, by)
        parts.append(_backward_diff(bones, fps, MAX_VELOCITY))   # (dbx, dby)

        # Scalars broadcast across joints. Every joint sees the same value, so
        # the graph convolution can combine a whole-body cue with local motion
        # instead of having to reconstruct it from the parts.
        angle = trunk_angle(xy)
        angle_rate = _backward_diff(angle[:, None], fps, MAX_VELOCITY)[:, 0]
        com = centre_of_mass(xy)
        com_rate = _backward_diff(com, fps, MAX_VELOCITY)

        scalars = np.stack([angle, angle_rate, com[:, 1], com_rate[:, 1]], axis=-1)
        parts.append(np.repeat(scalars[:, None, :], NUM_JOINTS, axis=1))

    stacked = np.concatenate(parts, axis=-1)              # [T, V, C]
    features = np.transpose(stacked, (2, 0, 1)).astype(np.float32)

    expected = channels_for(feature_set)
    if features.shape[0] != expected:
        raise AssertionError(
            f"{feature_set} produced {features.shape[0]} channels, expected {expected}"
        )
    return features
