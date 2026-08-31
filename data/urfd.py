"""UR Fall Detection adapter (Kwolek & Kepski) - the real-data benchmark.

UR Fall ships a **per-frame annotation**, which is what makes it usable here
without a re-annotation campaign:

    -1  the subject is upright / not lying
     0  transitional - the fall is in progress
     1  the subject is lying on the ground

`t*` is taken as the **first frame labelled 1**. Two things about that choice
matter enough to state plainly:

* It is **not derived from the skeleton the model consumes.** A `t*` inferred
  from joint kinematics would make lead time partly circular - the model would
  be scored against a target computed from its own inputs. This label comes from
  the dataset authors' own annotation of the RGB-D stream.
* It is a **conservative** proxy for ground contact. An annotator marks "lying"
  at or slightly after first contact, never before, so any error shortens the
  measured lead time rather than inflating it. The bias runs against the
  project's own claim, which is the direction a bias should run.

**ADL sequences also contain frames labelled 1** - 1470 of them - because
several activities involve deliberately lying down on a bed or the floor. Those
are *not* falls and are never labelled positive; they are the hardest negatives
in the set, and a model that fires on "person is horizontal" fails on them.

**Subject identity is not published per sequence.** UR Fall states the sequences
were performed by volunteers but does not map sequences to people, so a
subject-independent split is not constructible. Splits here are therefore
**sequence-independent**, which is a weaker guarantee, and results are reported
as a transfer check rather than as the primary benchmark (Section 13).
"""

from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# The dataset's own frame rate for the camera-0 RGB stream.
URFD_FPS = 30.0

LABEL_UPRIGHT = -1
LABEL_TRANSITION = 0
LABEL_LYING = 1


def read_labels(csv_path: str | Path) -> dict[str, dict[int, int]]:
    """Parse a `urfall-cam0-*.csv` into `{sequence: {frame: label}}`.

    Frame numbers in the CSV are 1-indexed; they are converted to 0-indexed here
    so they line up with the extracted image sequence.
    """
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or not row[0].strip():
                continue
            try:
                sequence, frame, label = row[0].strip(), int(row[1]), int(row[2])
            except ValueError:
                continue
            labels[sequence][frame - 1] = label
    return dict(labels)


def impact_frame(frame_labels: dict[int, int]) -> int | None:
    """First frame annotated as lying, or None if the subject never lies down."""
    lying = [f for f, label in frame_labels.items() if label == LABEL_LYING]
    return min(lying) if lying else None


def sequence_summary(labels: dict[str, dict[int, int]]) -> list[dict]:
    """Per-sequence frame counts and derived impact frames, for a sanity check."""
    out = []
    for name in sorted(labels):
        frames = labels[name]
        impact = impact_frame(frames)
        out.append({
            "sequence": name,
            "frames": len(frames),
            "impact_frame": impact,
            "lying_frames": sum(1 for v in frames.values() if v == LABEL_LYING),
            "is_fall": name.startswith("fall"),
        })
    return out


def list_zip_images(zip_path: str | Path) -> list[str]:
    """Image entries inside a sequence zip, in frame order.

    Names look like `fall-01-cam0-rgb-001.png`; sorting on the trailing number
    rather than lexically keeps frame 2 before frame 10.
    """
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist()
                 if n.lower().endswith((".png", ".jpg", ".jpeg")) and not n.endswith("/")]

    def frame_number(name: str) -> int:
        stem = Path(name).stem
        digits = ""
        for ch in reversed(stem):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        return int(digits) if digits else 0

    return sorted(names, key=frame_number)


def iter_zip_frames(zip_path: str | Path):
    """Yield `(index, BGR image)` for each frame, decoding one at a time.

    Streams rather than extracting: a sequence is ~60 MB of PNGs and there are
    70 of them, so unpacking them all to disk would cost 4 GB for no benefit.
    Nothing is written, which also keeps the privacy boundary intact - frames
    are decoded, posed and dropped.
    """
    import cv2

    with zipfile.ZipFile(zip_path) as archive:
        for index, name in enumerate(list_zip_images(zip_path)):
            buffer = np.frombuffer(archive.read(name), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is not None:
                yield index, image


def build_clip_metadata(
    sequence: str,
    num_frames: int,
    frame_labels: dict[int, int] | None,
) -> dict:
    """Metadata for a `ClipRecord` built from a UR Fall sequence."""
    is_fall = sequence.startswith("fall")
    impact = impact_frame(frame_labels or {}) if is_fall else None

    if is_fall and impact is not None:
        impact = min(impact, num_frames - 1)

    return {
        "clip_id": f"urfd_{sequence.replace('-', '')}",
        # No published subject mapping: the sequence is its own group, so splits
        # are sequence-independent rather than subject-independent.
        "subject": sequence,
        "label": "fall" if (is_fall and impact is not None) else "adl",
        "impact_frame": impact if (is_fall and impact is not None) else None,
        "activity": "fall" if is_fall else "adl",
        "view": "cam0",
        "source": "urfd",
        "fps": URFD_FPS,
        "meta": {
            "sequence": sequence,
            "lying_frames": sum(1 for v in (frame_labels or {}).values() if v == LABEL_LYING),
            "impact_from": "first frame annotated lying (urfall-cam0-*.csv)",
        },
    }
