"""Describe the real datasets, from the skeleton caches, for the website.

    python scripts/export_datasets.py

Writes `frontend/lib/datasets.json`. Every figure is computed from the cached
clips - nothing is typed into a component - so the Data pane cannot drift from
what the models were trained and tested on.

No torch: reads `.npz` metadata with numpy only. The phase counts therefore
re-implement the labelling arithmetic from `data/phases.py` instead of importing
it (importing `data` pulls in torch through the package `__init__`). The
duplication is guarded by `tests/test_datasets_export.py`, which compares this
function against the real labeller.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONF = 0.30
PHASES = ("normal", "imminent", "falling", "grounded")

# w_pre is 17 for BOTH datasets because that is what the combined benchmark trains
# and evaluates with (configs/combined.yaml). UR Fall's own config used 20 at 30 fps
# and Le2i's 17 at 25 fps; describing UR Fall with 20 here would report a negative
# time the combined results do not use.
SOURCES = {
    "urfd": {
        "name": "UR Fall Detection", "cache": "urfd", "w_pre": 17,
        "annotation": "First frame annotated 'lying' (dataset's own per-frame labels)",
        "onset_note": "Onset is a fixed 30 frames before impact on all 30 falls, so the falling "
                      "phase adds no information beyond the impact frame on this dataset.",
        "citation": "Kwolek & Kepski, UR Fall Detection Dataset",
    },
    "le2i": {
        "name": "Le2i / ImViA", "cache": "le2i", "w_pre": 17,
        "annotation": "Annotated fall END frame (dataset's own annotation)",
        "onset_note": "Onset-to-impact varies from 1 to 45 frames and is genuine annotation.",
        "citation": "Charfi et al., Le2i fall detection dataset (Home and Coffee room scenes)",
    },
}


def phase_counts(frames: int, impact: int | None, onset: int | None, w_pre: int) -> np.ndarray:
    """Frames per phase for one clip. Mirrors `data.phases.phase_labels`."""
    counts = np.zeros(4, dtype=np.int64)
    if impact is None:
        counts[0] = frames
        return counts
    if onset is None:
        start = max(0, impact - w_pre)
        counts[1] = impact + 1 - start
        counts[0] = start
    else:
        onset = int(np.clip(onset, 0, impact))
        pre = max(0, onset - w_pre)
        counts[1] = onset - pre
        counts[2] = impact + 1 - onset
        counts[0] = pre
    counts[3] = max(0, frames - (impact + 1))
    return counts


def describe(key: str, spec: dict) -> dict | None:
    cache = ROOT / "data" / "cache" / spec["cache"]
    paths = sorted(cache.glob("*.npz"))
    if not paths:
        return None

    clips, fps_seen = [], set()
    coverage_num = coverage_den = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            kp = handle["keypoints"]
            attrs = json.loads(str(handle["attrs"]))
        clips.append(attrs | {"frames": int(kp.shape[0])})
        fps_seen.add(round(float(attrs["fps"]), 2))
        coverage_num += int((kp[..., 2] >= CONF).sum())
        coverage_den += int(kp[..., 2].size)

    falls = [c for c in clips if c["label"] == "fall"]
    total_frames = sum(c["frames"] for c in clips)
    fps = float(np.mean([c["fps"] for c in clips]))

    phase = np.zeros(4, dtype=np.int64)
    negative_frames = 0
    for c in clips:
        phase += phase_counts(c["frames"], c.get("impact_frame"), c.get("onset_frame"), spec["w_pre"])
        # NEGATIVE time in the binary sense the metrics use (evaluation/metrics.py):
        # every frame of an ADL clip, and for a fall only the frames before the
        # imminent window. This is larger than the phase scheme's 'normal', which
        # also reserves the pre-onset window - and it is the denominator every
        # false-alarm rate on the site is computed over.
        negative_frames += (c["frames"] if c.get("impact_frame") is None
                            else max(0, int(c["impact_frame"]) - spec["w_pre"]))

    pre_impact = [c["impact_frame"] for c in falls if c.get("impact_frame") is not None]
    onset_gap = [c["impact_frame"] - c["onset_frame"] for c in falls
                 if c.get("impact_frame") is not None and c.get("onset_frame") is not None]

    return {
        "id": key, "name": spec["name"], "citation": spec["citation"],
        "clips": len(clips), "falls": len(falls), "adls": len(clips) - len(falls),
        "minutes": round(total_frames / fps / 60, 1),
        "fps": sorted(fps_seen),
        "negative_minutes": round(negative_frames / fps / 60, 1),
        "pose_coverage": round(coverage_num / max(coverage_den, 1), 4),
        "annotation": spec["annotation"], "onset_note": spec["onset_note"],
        "pre_impact_frames": ({"min": int(min(pre_impact)), "median": int(np.median(pre_impact)),
                               "max": int(max(pre_impact))} if pre_impact else None),
        "onset_to_impact_frames": ({"min": int(min(onset_gap)), "median": int(np.median(onset_gap)),
                                    "max": int(max(onset_gap))} if onset_gap else None),
        "phase_frames": {n: int(v) for n, v in zip(PHASES, phase)},
        "w_pre_frames": spec["w_pre"],
    }


def main() -> int:
    datasets = [d for d in (describe(k, s) for k, s in SOURCES.items()) if d]
    combined = None
    if len(datasets) == len(SOURCES):
        phase = {n: sum(d["phase_frames"][n] for d in datasets) for n in PHASES}
        combined = {
            "clips": sum(d["clips"] for d in datasets), "falls": sum(d["falls"] for d in datasets),
            "adls": sum(d["adls"] for d in datasets),
            "minutes": round(sum(d["minutes"] for d in datasets), 1),
            "negative_minutes": round(sum(d["negative_minutes"] for d in datasets), 1),
            "phase_frames": phase,
        }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets, "combined": combined,
        "limits": [
            "Falls are ACTED by volunteers - not recorded elderly falls.",
            "Neither dataset publishes actor identity, so splits are clip-independent, "
            "not subject-independent, and may overstate performance on unseen people.",
            "Pose coverage drops after impact: the estimator struggles with a person on the floor.",
        ],
    }
    out = ROOT / "frontend" / "lib" / "datasets.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for d in datasets:
        print(f"  {d['name']}: {d['clips']} clips, {d['falls']} falls, {d['minutes']} min, "
              f"coverage {d['pose_coverage']:.1%}")
    if combined:
        print(f"  combined: {combined['clips']} clips, {combined['falls']} falls, "
              f"{combined['negative_minutes']} min negative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
