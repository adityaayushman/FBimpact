"""Skeleton graph adjacency for the graph convolution.

Implements the three partitioning strategies from ST-GCN (Yan et al., AAAI 2018).
The `spatial` strategy is the default and the one the paper reports: it splits a
joint's neighbourhood into the joint itself, neighbours closer to the body centre
(centripetal) and neighbours further from it (centrifugal). That split is what
lets a graph convolution distinguish "the trunk pulled the hip over" from "the
hip pulled the trunk over", which is precisely the distinction a fall
explanation needs to make.
"""

from __future__ import annotations

import numpy as np

from data.skeleton import CENTRE_JOINTS, NUM_JOINTS, adjacency_matrix, hop_distance

STRATEGIES = ("uniform", "distance", "spatial")


def normalise_adjacency(a: np.ndarray) -> np.ndarray:
    """Row-normalise `A` so each node's incoming weights sum to one."""
    degree = a.sum(axis=1)
    inv = np.zeros_like(degree)
    np.divide(1.0, degree, out=inv, where=degree > 0)
    return (a * inv[:, None]).astype(np.float32)


def centre_hops() -> np.ndarray:
    """Hop distance from every joint to the body centre.

    The centre is the hip pair rather than a single joint, so the left/right
    symmetry of the skeleton is preserved and mirrored clips get a mirrored
    partition rather than a different one.
    """
    hops = hop_distance(max_hop=NUM_JOINTS)
    return hops[:, list(CENTRE_JOINTS)].min(axis=1)


def build_adjacency(strategy: str = "spatial", max_hop: int = 1) -> np.ndarray:
    """Stacked adjacency `A [K, V, V]` for the chosen partitioning.

    Args:
        strategy: `"uniform"` (K=1), `"distance"` (K=max_hop+1) or `"spatial"` (K=3).
        max_hop: neighbourhood radius in graph hops.

    Returns:
        `[K, V, V]` float32, each slice row-normalised.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    hops = hop_distance(max_hop=max_hop)
    valid = [h for h in range(max_hop + 1)]
    neighbourhood = np.isin(hops, valid).astype(np.float32)

    if strategy == "uniform":
        return normalise_adjacency(neighbourhood)[None]

    if strategy == "distance":
        stack = [np.where(hops == h, 1.0, 0.0).astype(np.float32) for h in valid]
        return np.stack([normalise_adjacency(s) for s in stack])

    # spatial
    centre = centre_hops()
    normalised = normalise_adjacency(neighbourhood)
    root = np.zeros_like(normalised)
    centripetal = np.zeros_like(normalised)
    centrifugal = np.zeros_like(normalised)

    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if neighbourhood[i, j] == 0:
                continue
            # `normalised[i, j]` is the weight of j's contribution to i.
            if centre[j] == centre[i]:
                root[i, j] = normalised[i, j]
            elif centre[j] < centre[i]:
                centripetal[i, j] = normalised[i, j]
            else:
                centrifugal[i, j] = normalised[i, j]

    return np.stack([root, centripetal, centrifugal]).astype(np.float32)


class Graph:
    """Container that keeps the adjacency and its provenance together."""

    def __init__(self, strategy: str = "spatial", max_hop: int = 1) -> None:
        self.strategy = strategy
        self.max_hop = max_hop
        self.A = build_adjacency(strategy, max_hop)
        self.base = adjacency_matrix(self_loops=False)

    @property
    def num_subsets(self) -> int:
        return int(self.A.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.A.shape[-1])

    def __repr__(self) -> str:
        return (
            f"Graph(strategy={self.strategy!r}, max_hop={self.max_hop}, "
            f"K={self.num_subsets}, V={self.num_joints})"
        )
