"""The cached clip - the only artefact that crosses the privacy boundary.

Everything upstream of this file touches pixels; everything downstream sees a
`ClipRecord` and nothing else. A cached clip holds joint coordinates,
confidences and the metadata needed to split and label it. It never holds an
image, a frame, or a path that can be decoded back into one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .skeleton import NUM_JOINTS

CACHE_SUFFIX = ".npz"


@dataclass
class ClipRecord:
    """One person's skeleton sequence from one video clip."""

    clip_id: str
    """Unique identifier, used as the cache filename."""

    subject: str
    """Subject identifier - the grouping key for subject-independent splits."""

    keypoints: np.ndarray
    """`[T, V, 3]` of `(x, y, confidence)` in pixel coordinates."""

    fps: float = 30.0

    label: str = "adl"
    """`"fall"` or `"adl"`."""

    impact_frame: int | None = None
    """`t*`, first frame of ground contact. Required when `label == "fall"`."""

    activity: str = "unknown"
    """Dataset-specific activity name, kept for per-activity error analysis."""

    view: str = "default"
    """Camera identifier, kept so per-view results can be reported (Section 18)."""

    source: str = "unknown"
    """Dataset name, so cross-dataset transfer runs stay traceable."""

    meta: dict = field(default_factory=dict)
    """Anything else the adapter wants to keep; must be JSON-serialisable."""

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.keypoints.ndim != 3 or self.keypoints.shape[1:] != (NUM_JOINTS, 3):
            raise ValueError(
                f"{self.clip_id}: keypoints must be [T, {NUM_JOINTS}, 3], "
                f"got {self.keypoints.shape}"
            )
        if self.label not in ("fall", "adl"):
            raise ValueError(f"{self.clip_id}: label must be 'fall' or 'adl', got {self.label!r}")
        if self.label == "fall":
            if self.impact_frame is None:
                raise ValueError(f"{self.clip_id}: a fall clip needs an impact_frame")
            if not 0 <= self.impact_frame < self.num_frames:
                raise ValueError(
                    f"{self.clip_id}: impact_frame {self.impact_frame} outside "
                    f"clip of {self.num_frames} frames"
                )
        else:
            self.impact_frame = None

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def is_fall(self) -> bool:
        return self.label == "fall"

    @property
    def duration_s(self) -> float:
        return self.num_frames / float(self.fps)

    def save(self, directory: str | Path) -> Path:
        """Write the clip to `<directory>/<clip_id>.npz`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.clip_id}{CACHE_SUFFIX}"
        attrs = {k: v for k, v in asdict(self).items() if k != "keypoints"}
        np.savez_compressed(
            path, keypoints=self.keypoints, attrs=json.dumps(attrs)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ClipRecord":
        """Read a clip written by `save`."""
        with np.load(path, allow_pickle=False) as handle:
            keypoints = handle["keypoints"]
            attrs = json.loads(str(handle["attrs"]))
        return cls(keypoints=keypoints, **attrs)


def load_cache(directory: str | Path) -> list[ClipRecord]:
    """Load every cached clip in a directory, sorted by clip id."""
    directory = Path(directory)
    paths = sorted(directory.glob(f"*{CACHE_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"no cached clips in {directory}. Populate it with "
            f"`python scripts/make_synthetic.py` or `python scripts/cache_poses.py`."
        )
    return [ClipRecord.load(p) for p in paths]


def summarise(clips: list[ClipRecord]) -> dict:
    """Counts used in the dataset table and in run logs."""
    falls = [c for c in clips if c.is_fall]
    adls = [c for c in clips if not c.is_fall]
    return {
        "clips": len(clips),
        "falls": len(falls),
        "adls": len(adls),
        "subjects": len({c.subject for c in clips}),
        "views": len({c.view for c in clips}),
        "adl_hours": sum(c.duration_s for c in adls) / 3600.0,
        "total_hours": sum(c.duration_s for c in clips) / 3600.0,
    }
