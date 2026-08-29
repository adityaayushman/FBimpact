"""Section 14 augmentation, applied to normalised coordinates.

All of these operate on `xy [T, V, 2]` *before* velocity is differenced, so the
velocity channel stays consistent with the positions it was derived from. The
one exception is temporal jitter, which acts on window start indices and lives
in the dataset.
"""

from __future__ import annotations

import numpy as np

from .skeleton import flip_permutation


class Augmenter:
    """Composable geometric augmentation for normalised skeleton clips."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        scale_range: tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        rotate_deg: float = 0.0,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            flip_prob: probability of mirroring horizontally (with a left/right
                joint swap, otherwise the skeleton becomes anatomically absurd).
            scale_range: uniform range for an isotropic scale.
            noise_std: standard deviation of Gaussian joint noise, in torso
                lengths - 0.01 is roughly a centimetre of jitter on an adult.
            rotate_deg: maximum in-plane rotation. Off by default: a fall is
                defined relative to gravity, and rotating the skeleton destroys
                the vertical reference the task depends on.
            enabled: set False for validation and test.
        """
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.rotate_deg = rotate_deg
        self.enabled = enabled
        self._perm = flip_permutation()

    def __call__(self, xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augment `xy [T, V, 2]`, returning a new array."""
        if not self.enabled:
            return xy
        xy = np.asarray(xy, dtype=np.float32).copy()

        if rng.random() < self.flip_prob:
            xy[..., 0] *= -1.0
            xy = xy[:, self._perm, :]

        if self.scale_range is not None:
            lo, hi = self.scale_range
            if hi > lo:
                xy *= float(rng.uniform(lo, hi))

        if self.rotate_deg > 0.0:
            theta = np.deg2rad(rng.uniform(-self.rotate_deg, self.rotate_deg))
            cos, sin = np.cos(theta), np.sin(theta)
            rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
            xy = xy @ rotation.T

        if self.noise_std > 0.0:
            xy += rng.normal(0.0, self.noise_std, size=xy.shape).astype(np.float32)

        return xy.astype(np.float32)

    @classmethod
    def from_config(cls, cfg: dict | None, enabled: bool = True) -> "Augmenter":
        cfg = dict(cfg or {})
        scale = cfg.get("scale_range", (0.9, 1.1))
        return cls(
            flip_prob=float(cfg.get("flip_prob", 0.5)),
            scale_range=tuple(scale) if scale else None,
            noise_std=float(cfg.get("noise_std", 0.01)),
            rotate_deg=float(cfg.get("rotate_deg", 0.0)),
            enabled=enabled and bool(cfg.get("enabled", True)),
        )
