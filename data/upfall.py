"""UP-Fall adapter (Section 13, primary benchmark).

UP-Fall ships 17 subjects x 11 activities x 3 trials x 2 cameras. Activities 1-5
are falls, 6-11 are activities of daily living:

    1 fall forward (hands)   6  walking      9  picking up an object
    2 fall forward (knees)   7  standing     10 jumping
    3 fall backwards         8  sitting      11 laying
    4 fall sideward
    5 fall sitting in an empty chair

What UP-Fall does **not** ship is an impact frame. Its labels are activity
intervals, not the instant of ground contact, and every lead-time number in this
project is measured from that instant. So `t*` has to be annotated, which is why
Section 5 lists the onset-annotation protocol as a deliverable rather than an
afterthought: run `scripts/annotate_onsets.py`, which writes the `onsets.csv`
this adapter reads.

Clips whose `t*` has not been annotated are still loaded as *unlabelled falls*
and excluded from both training and evaluation by default, so a partially
annotated cache degrades to a smaller experiment rather than to a wrong one.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .clips import ClipRecord

FALL_ACTIVITIES: dict[int, str] = {
    1: "fall_forward_hands",
    2: "fall_forward_knees",
    3: "fall_backward",
    4: "fall_sideward",
    5: "fall_sitting_empty_chair",
}

ADL_ACTIVITIES: dict[int, str] = {
    6: "walking",
    7: "standing",
    8: "sitting",
    9: "picking_up",
    10: "jumping",
    11: "laying",
}

ACTIVITIES = {**FALL_ACTIVITIES, **ADL_ACTIVITIES}

# Matches the canonical UP-Fall naming, e.g. "Subject1Activity3Trial2Camera1".
_NAME_RE = re.compile(
    r"Subject(?P<subject>\d+)Activity(?P<activity>\d+)Trial(?P<trial>\d+)Camera(?P<camera>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UpFallKey:
    """The four fields that identify a UP-Fall recording."""

    subject: int
    activity: int
    trial: int
    camera: int

    @property
    def clip_id(self) -> str:
        return (
            f"upfall_s{self.subject:02d}_a{self.activity:02d}"
            f"_t{self.trial:d}_c{self.camera:d}"
        )

    @property
    def is_fall(self) -> bool:
        return self.activity in FALL_ACTIVITIES

    @property
    def activity_name(self) -> str:
        return ACTIVITIES.get(self.activity, f"activity_{self.activity}")


def parse_name(name: str) -> UpFallKey | None:
    """Extract a `UpFallKey` from a UP-Fall file or directory name."""
    match = _NAME_RE.search(str(name))
    if not match:
        return None
    return UpFallKey(
        subject=int(match["subject"]),
        activity=int(match["activity"]),
        trial=int(match["trial"]),
        camera=int(match["camera"]),
    )


def find_videos(root: str | Path, patterns: tuple[str, ...] = ("*.mp4", "*.avi", "*.zip")) -> list[Path]:
    """Every UP-Fall recording under `root` whose name parses."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"UP-Fall root not found: {root}")
    found = [p for pattern in patterns for p in root.rglob(pattern)]
    return sorted(p for p in found if parse_name(p.name) is not None)


def read_onsets(path: str | Path) -> dict[str, int]:
    """Read `onsets.csv` -> `{clip_id: impact_frame}`.

    Expected columns: `clip_id`, `impact_frame`, and optionally `annotator` and
    `notes`. Rows with an empty or negative `impact_frame` are treated as
    not-yet-annotated and omitted.
    """
    path = Path(path)
    if not path.exists():
        return {}
    onsets: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("impact_frame") or "").strip()
            if not raw:
                continue
            try:
                frame = int(float(raw))
            except ValueError:
                continue
            if frame >= 0:
                onsets[row["clip_id"].strip()] = frame
    return onsets


def write_onsets(path: str | Path, onsets: dict[str, int], annotator: str = "") -> Path:
    """Write an `onsets.csv`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "impact_frame", "annotator", "notes"])
        for clip_id in sorted(onsets):
            writer.writerow([clip_id, onsets[clip_id], annotator, ""])
    return path


def attach_labels(
    clips: list[ClipRecord],
    onsets: dict[str, int],
    drop_unannotated_falls: bool = True,
) -> tuple[list[ClipRecord], list[str]]:
    """Attach impact frames from `onsets` to cached UP-Fall clips.

    Args:
        clips: clips loaded from the pose cache, with `label` already set from
            the activity number but `impact_frame` still unknown.
        onsets: `{clip_id: impact_frame}` from `read_onsets`.
        drop_unannotated_falls: when True (the default) a fall with no annotated
            `t*` is removed rather than guessed at. Guessing would corrupt the
            headline metric silently; dropping only shrinks the sample.

    Returns:
        `(usable_clips, dropped_clip_ids)`.
    """
    usable: list[ClipRecord] = []
    dropped: list[str] = []

    for clip in clips:
        if not clip.is_fall:
            usable.append(clip)
            continue
        frame = onsets.get(clip.clip_id)
        if frame is None or not 0 <= frame < clip.num_frames:
            if drop_unannotated_falls:
                dropped.append(clip.clip_id)
                continue
            raise ValueError(
                f"{clip.clip_id}: no valid impact_frame in onsets.csv "
                f"(clip has {clip.num_frames} frames)"
            )
        clip.impact_frame = int(frame)
        usable.append(clip)

    return usable, dropped


def clip_record_stub(key: UpFallKey, keypoints, fps: float = 18.0) -> ClipRecord:
    """Build a `ClipRecord` for a UP-Fall recording, before `t*` is known.

    UP-Fall's cameras record at roughly 18 fps rather than 30, which matters:
    lead time is reported in seconds, so the frame rate has to come from the
    dataset and not from a default. Falls are created with a placeholder
    `impact_frame` of 0 and corrected by `attach_labels`.
    """
    return ClipRecord(
        clip_id=key.clip_id,
        subject=f"S{key.subject:02d}",
        keypoints=keypoints,
        fps=fps,
        label="fall" if key.is_fall else "adl",
        impact_frame=0 if key.is_fall else None,
        activity=key.activity_name,
        view=f"cam{key.camera}",
        source="upfall",
        meta={"trial": key.trial, "activity_id": key.activity},
    )
