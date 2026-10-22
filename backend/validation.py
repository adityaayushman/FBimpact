"""Input validation for the inference API.

Pure NumPy and no torch, so it is testable on a machine whose commit is taken by
a training job - and so that a malformed request is rejected before the 400 MB
torch workspace is touched.

Every input maps to exactly one outcome. The checks run in a fixed order so two
rules can never both claim a request:

    1. structure   shape, element types, size
    2. per frame   which frames are missing
    3. per clip    how much is missing, and how long a run
    4. range       coordinates and confidences on the frames that remain

The policy on dropout is the one that matters in practice. A pose estimator drops
a person for a few frames when they are occluded or fall out of shot, and a
fall is *exactly* when that happens. Rejecting any clip with a null in it would
make the service unusable on the one event it exists for; silently interpolating
across a long gap would invent motion the person never made. So short, sparse
gaps are bridged and flagged, and anything longer or more frequent is refused
with the numbers that caused the refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_JOINTS = 17

MIN_FPS, MAX_FPS = 5.0, 120.0

# At most this fraction of frames may be missing, and no gap may be longer than
# this many consecutive frames, for the clip to be repaired rather than refused.
MAX_MISSING_FRACTION = 0.20
MAX_GAP_FRAMES = 10

# A sanity bound on pixel coordinates, not a learned range. The model normalises
# by torso length, so it is scale-invariant and any frame size is legitimate; this
# only catches values that cannot be pixels at all (8K video is 7680 wide).
MAX_ABS_COORD = 50_000.0


class ApiError(Exception):
    """An error with a stable machine-readable code and an HTTP status."""

    def __init__(self, status: int, code: str, message: str, **extra) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra

    def body(self, request_id: str | None = None) -> dict:
        err = {"code": self.code, "message": self.message, **self.extra}
        if request_id:
            err["request_id"] = request_id
        return {"error": err}


@dataclass
class Prepared:
    """A validated clip, ready for the model."""

    keypoints: np.ndarray
    """`[T, 17, 3]` float32, missing frames zeroed with zero confidence so the
    existing normalisation interpolates across them per joint."""

    no_person: bool = False
    """True when every frame was missing. Not an error: nobody is in shot."""

    degraded: bool = False
    interpolated_frames: list[int] = field(default_factory=list)
    clipped_confidence: bool = False


def _missing_frames(kp: np.ndarray) -> np.ndarray:
    """Boolean `[T]`: a frame is missing when no joint has a usable observation.

    A joint is unusable if its coordinates are non-finite or its confidence is
    non-finite or not positive. An all-zero frame is the pose-estimator
    convention for "nobody detected" and has confidence 0, so it is caught by
    the same rule rather than needing a special case.
    """
    finite_xy = np.isfinite(kp[..., 0]) & np.isfinite(kp[..., 1])
    conf = kp[..., 2]
    usable = finite_xy & np.isfinite(conf) & (conf > 0.0)
    return ~usable.any(axis=1)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def validate(
    raw,
    fps: float,
    min_frames: int,
    max_frames: int,
    impact_frame: int | None = None,
) -> Prepared:
    """Validate and repair a request's keypoints.

    Args:
        raw: the decoded JSON value for `keypoints`; `None` entries are allowed
            and mean "missing".
        fps: frame rate of the clip.
        min_frames: the model's window; shorter clips are refused, never padded.
        max_frames: the service's frame cap.
        impact_frame: `t*` if the caller supplied one.

    Raises:
        ApiError: with the status and code documented in `DEPLOY.md`.
    """
    # -- 1. structure --------------------------------------------------------
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ApiError(422, "bad_shape", "keypoints must be a list of frames, each 17 joints of [x, y, confidence]")

    if len(raw) == 0:
        raise ApiError(422, "empty_clip", "keypoints is empty: send at least one frame")

    try:
        kp = np.array(raw, dtype=np.float64)   # None -> nan, which `_missing_frames` handles
    except (TypeError, ValueError):
        # np.array raises ValueError on a ragged list and TypeError on strings or
        # objects; both mean the structure is not a clean numeric cube.
        if _ragged(raw):
            raise ApiError(422, "bad_shape", "every frame must have exactly 17 joints of 3 numbers") from None
        raise ApiError(422, "non_numeric", "keypoints must contain only numbers (or null for a missing value)") from None

    if kp.ndim != 3 or kp.shape[1:] != (NUM_JOINTS, 3):
        raise ApiError(422, "bad_shape",
                       f"expected shape [T, {NUM_JOINTS}, 3], got {list(kp.shape)}",
                       got=list(kp.shape))

    if not (MIN_FPS <= fps <= MAX_FPS) or not np.isfinite(fps):
        raise ApiError(422, "bad_fps", f"fps must be between {MIN_FPS:g} and {MAX_FPS:g}, got {fps}")

    frames = kp.shape[0]
    if frames < min_frames:
        raise ApiError(422, "too_short",
                       f"need at least {min_frames} frames (the model's window), got {frames}",
                       minimum=min_frames, got=frames)
    if frames > max_frames:
        raise ApiError(413, "too_long",
                       f"{frames} frames exceeds the {max_frames}-frame limit (~{max_frames / fps:.0f}s at {fps:g} fps)",
                       maximum=max_frames, got=frames)

    if impact_frame is not None and not 0 <= impact_frame < frames:
        raise ApiError(422, "bad_impact_frame", f"impact_frame {impact_frame} is outside a {frames}-frame clip")

    # -- 2. per frame --------------------------------------------------------
    missing = _missing_frames(kp)

    if bool(missing.all()):
        return Prepared(keypoints=np.zeros((frames, NUM_JOINTS, 3), dtype=np.float32), no_person=True)

    # -- 3. per clip ---------------------------------------------------------
    fraction = float(missing.mean())
    longest = _longest_run(missing)
    if fraction > MAX_MISSING_FRACTION or longest > MAX_GAP_FRAMES:
        raise ApiError(
            422, "too_much_missing",
            f"{fraction:.0%} of frames are missing (limit {MAX_MISSING_FRACTION:.0%}) and the longest "
            f"gap is {longest} frames (limit {MAX_GAP_FRAMES}); too little observed motion to analyse",
            missing_fraction=round(fraction, 4), longest_gap=longest,
            max_fraction=MAX_MISSING_FRACTION, max_gap=MAX_GAP_FRAMES,
        )

    # -- 4. range ------------------------------------------------------------
    observed = ~missing
    coords = kp[observed][..., :2]
    finite = coords[np.isfinite(coords)]
    if finite.size and float(np.abs(finite).max()) > MAX_ABS_COORD:
        raise ApiError(422, "out_of_range",
                       f"coordinates must be pixel values within +/-{MAX_ABS_COORD:g}; "
                       f"got a magnitude of {float(np.abs(finite).max()):g}",
                       limit=MAX_ABS_COORD)

    # -- repair --------------------------------------------------------------
    out = kp.copy()
    conf = out[..., 2]
    in_range = np.isfinite(conf)
    clipped = bool(((conf[in_range] < 0.0) | (conf[in_range] > 1.0)).any())
    out[..., 2] = np.where(np.isfinite(conf), np.clip(conf, 0.0, 1.0), 0.0)
    # Unusable coordinates become zero at zero confidence. That is the existing
    # pose-estimator convention and `data.normalize` interpolates across it per
    # joint, so no separate interpolation code exists here to drift from training.
    bad_xy = ~np.isfinite(out[..., 0]) | ~np.isfinite(out[..., 1])
    out[..., 0] = np.where(bad_xy, 0.0, out[..., 0])
    out[..., 1] = np.where(bad_xy, 0.0, out[..., 1])
    out[..., 2] = np.where(bad_xy, 0.0, out[..., 2])
    out[missing] = 0.0

    gaps = np.flatnonzero(missing).tolist()
    return Prepared(
        keypoints=out.astype(np.float32),
        degraded=bool(gaps),
        interpolated_frames=gaps,
        clipped_confidence=clipped,
    )


def _ragged(raw) -> bool:
    """True when frames or joints have inconsistent lengths."""
    try:
        frame_lengths = {len(f) for f in raw}
        if frame_lengths != {NUM_JOINTS}:
            return True
        return any(len(j) != 3 for f in raw for j in f)
    except TypeError:
        return False
