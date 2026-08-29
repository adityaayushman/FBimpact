"""COCO-17 skeleton topology.

The single source of truth for joint ordering, graph edges and left/right
mirroring. Everything downstream (normalisation, the graph convolution
adjacency, augmentation, joint-level explanations) reads its layout from here so
a change of pose estimator only has to be reflected in one place.
"""

from __future__ import annotations

import numpy as np

# Joint order as emitted by COCO-trained 2D pose estimators (RTMPose, YOLO-Pose,
# AlphaPose all use this ordering).
JOINT_NAMES: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

NUM_JOINTS = len(JOINT_NAMES)

JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# Anatomical bones, as (parent, child) index pairs. Used to build the graph
# convolution adjacency; the graph is undirected, the direction only defines the
# centrifugal/centripetal partition in `models.graph`.
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
)

# Joints swapped when a clip is mirrored horizontally.
FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
)

LEFT_HIP = JOINT_INDEX["left_hip"]
RIGHT_HIP = JOINT_INDEX["right_hip"]
LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]

# The body centre used as the coordinate origin during normalisation. Averaging
# the two hips is more stable than any single joint.
CENTRE_JOINTS: tuple[int, ...] = (LEFT_HIP, RIGHT_HIP)

# The two ends of the torso, whose distance sets the per-frame scale.
TORSO_TOP_JOINTS: tuple[int, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER)
TORSO_BOTTOM_JOINTS: tuple[int, ...] = (LEFT_HIP, RIGHT_HIP)

# Coarse anatomical groups, used only to render human-readable explanations
# ("trunk lean", "hip drop", "knee buckling") from a ranked joint list.
JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4),
    "trunk": (5, 6, 11, 12),
    "arms": (7, 8, 9, 10),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}


def adjacency_matrix(self_loops: bool = True) -> np.ndarray:
    """Binary undirected adjacency `A [V, V]` over the skeleton graph."""
    a = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in BONES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    if self_loops:
        a[np.arange(NUM_JOINTS), np.arange(NUM_JOINTS)] = 1.0
    return a


def hop_distance(max_hop: int = 1) -> np.ndarray:
    """Shortest-path hop count between joints, `inf` beyond `max_hop`."""
    a = adjacency_matrix(self_loops=False)
    hops = np.full((NUM_JOINTS, NUM_JOINTS), np.inf, dtype=np.float32)
    # reach[k] is non-zero wherever a path of length exactly <= k exists
    reach = [np.linalg.matrix_power(a, k) for k in range(max_hop + 1)]
    for k in range(max_hop, -1, -1):
        hops[reach[k] > 0] = k
    return hops


def flip_permutation() -> np.ndarray:
    """Index permutation that swaps left and right joints."""
    perm = np.arange(NUM_JOINTS)
    for i, j in FLIP_PAIRS:
        perm[i], perm[j] = j, i
    return perm


def group_of(joint: int) -> str:
    """Coarse anatomical group a joint belongs to (first match wins)."""
    for name, members in JOINT_GROUPS.items():
        if joint in members and name not in ("hips",):
            return name
    return "trunk"
