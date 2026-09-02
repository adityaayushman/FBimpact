"""Le2i / ImViA fall dataset adapter.

Four scenes were recorded, but only **Home** and **Coffee room** ship annotation
files. Only those two are usable here: without an annotation there is no impact
frame, and without an impact frame there is no lead time to measure. The Office
and Lecture room subsets are ignored rather than guessed at.

Annotation format (one `video (N).txt` per clip):

    line 1        frame at which the fall STARTS
    line 2        frame at which the fall ENDS
    lines 3..n    per-frame bounding boxes, `frame,x1,y1,x2,y2`

`0` on both of the first two lines means the clip contains no fall.

`t*` is taken as the **end** frame. The annotators marked the interval over
which the fall takes place, so its end is the moment the body has finished
arriving at the floor - the same quantity UR Fall's "first frame lying" gives,
and the same reasoning applies twice over: it is not derived from the skeleton
the model consumes, and it lands at or after true first contact, so any error
shortens measured lead time rather than inflating it.

Le2i publishes no subject identity either, so splits are **clip-independent**,
which is weaker than subject-independent. Stated wherever the results appear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# Scenes with annotation files. Names as they appear in the Kaggle mirror's
# directory tree, matched case-insensitively.
ANNOTATED_SCENES = ("coffee_room", "home")

# Le2i cameras record at 25 fps, not 30. Lead time is reported in seconds, so
# using the wrong rate would rescale every number in the results table.
LE2I_FPS = 25.0


@dataclass(frozen=True)
class Le2iAnnotation:
    """Parsed contents of one `video (N).txt`."""

    start: int | None
    """0-indexed frame at which the fall begins, or None when there is no fall."""

    end: int | None
    """0-indexed frame at which the fall ends - the impact frame."""

    boxes: int
    """How many per-frame bounding boxes were listed, i.e. the annotated length."""

    @property
    def has_fall(self) -> bool:
        return self.start is not None and self.end is not None


def parse_annotation(text: str) -> Le2iAnnotation:
    """Parse an annotation file's contents.

    Tolerant of blank lines and of the trailing box list being absent, because
    the mirrors differ slightly in whitespace and line endings.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return Le2iAnnotation(None, None, 0)

    def first_int(line: str) -> int | None:
        match = re.match(r"^-?\d+", line)
        return int(match.group()) if match else None

    start, end = first_int(lines[0]), first_int(lines[1])
    boxes = sum(1 for ln in lines[2:] if "," in ln)

    # 0/0 is the dataset's way of saying "no fall in this clip".
    if not start or not end or end < start:
        return Le2iAnnotation(None, None, boxes)

    # The file is 1-indexed; everything downstream is 0-indexed.
    return Le2iAnnotation(start - 1, end - 1, boxes)


def scene_of(path: str) -> str | None:
    """Scene name for an archive path, or None if it is not an annotated scene."""
    lowered = str(path).lower()
    for scene in ANNOTATED_SCENES:
        if scene in lowered:
            return scene
    return None


def clip_key(path: str) -> str | None:
    """Stable identifier from an archive path, e.g. `coffee_room_01_video07`.

    Le2i names files `video (7).avi` with a space and parentheses, which makes a
    poor cache filename and an ambiguous sort key, so the number is extracted
    and zero-padded.
    """
    pure = PurePosixPath(str(path).replace("\\", "/"))
    scene = scene_of(str(pure))
    if scene is None:
        return None

    match = re.search(r"(\d+)", pure.stem)
    if not match:
        return None

    # Keep the scene's sub-folder index (Coffee_room_01 vs _02) when present.
    variant = ""
    for part in pure.parts:
        found = re.fullmatch(rf"(?i){scene}_(\d+)", part)
        if found:
            variant = f"_{int(found.group(1)):02d}"
            break

    return f"{scene}{variant}_video{int(match.group(1)):02d}"


def build_clip_metadata(key: str, annotation: Le2iAnnotation, num_frames: int) -> dict:
    """Metadata for a `ClipRecord` built from a Le2i clip."""
    impact = annotation.end if annotation.has_fall else None
    if impact is not None:
        impact = min(impact, num_frames - 1)

    scene = key.split("_video")[0]
    return {
        "clip_id": f"le2i_{key}",
        # No published subject identity: the clip is its own group, so splits
        # are clip-independent rather than subject-independent.
        "subject": key,
        "label": "fall" if impact is not None else "adl",
        "impact_frame": impact,
        "activity": scene.replace("_", " "),
        "view": scene,
        "source": "le2i",
        "fps": LE2I_FPS,
        "meta": {
            "scene": scene,
            "fall_start": annotation.start,
            "fall_end": annotation.end,
            "impact_from": "annotated fall end frame",
        },
    }
