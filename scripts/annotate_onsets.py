"""The fall-onset annotation protocol (Section 5, secondary contribution).

    python scripts/annotate_onsets.py --cache data/cache/upfall --out data/annotations/upfall_onsets.csv
    python scripts/annotate_onsets.py --cache data/cache/upfall --suggest --out ...
    python scripts/annotate_onsets.py --agreement a.csv b.csv

`t*` is the **first frame of ground contact**: the first frame in which any part
of the body that was not previously supporting weight is in contact with the
floor. Everything the project claims is measured from it, so the protocol has to
be written down, followed, and its reliability reported - which is what
`--agreement` is for.

--------------------------------------------------------------------------------
PROTOCOL
--------------------------------------------------------------------------------
1. Two annotators label every fall clip independently. They do not see each
   other's labels, and they do not see the model's predictions.
2. For each clip, step frame by frame and mark the first frame at which ground
   contact has *clearly* occurred. When it falls between two frames, take the
   later one - a systematically late `t*` shortens the measured lead time, so
   the bias runs against the project's own claim rather than in favour of it.
3. Record uncertainty in the `notes` column ("occluded", "off-frame",
   "ambiguous") rather than guessing. Clips marked ambiguous by either annotator
   are excluded and the exclusion is reported.
4. Run `--agreement` and report the mean absolute difference in frames and
   seconds, plus the proportion within one frame, in the paper (Section 18:
   "fall-onset timing is subjective").
5. Resolve disagreements above the tolerance by joint review; take the mean for
   the rest.

`--suggest` fills the column with a kinematic heuristic. It is a *starting
point for human review, never an annotation*: a heuristic-labelled `t*` derived
from the same skeleton the model consumes would make lead time partly circular.
Suggested rows are written with `annotator = "heuristic"` so unreviewed rows can
always be found again.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.clips import ClipRecord, load_cache  # noqa: E402
from data.skeleton import CENTRE_JOINTS  # noqa: E402
from data.upfall import read_onsets  # noqa: E402


def suggest_impact_frame(clip: ClipRecord) -> tuple[int, float]:
    """Kinematic guess at ground contact, plus a confidence in `[0, 1]`.

    Finds the fastest downward motion of the mid-hip, then walks forward to the
    first frame where that descent has essentially stopped - the body arriving
    at the floor. Confidence is the descent's prominence relative to the rest of
    the clip, so a clip where nothing much happens scores low and gets a human's
    attention first.
    """
    hips = clip.keypoints[:, list(CENTRE_JOINTS), :2].mean(axis=1)   # [T, 2]
    y = hips[:, 1]                                                    # image y grows downward
    if len(y) < 5:
        return max(len(y) - 1, 0), 0.0

    # Light smoothing: pose noise otherwise dominates the derivative.
    kernel = np.ones(3) / 3.0
    y = np.convolve(y, kernel, mode="same")
    velocity = np.diff(y, prepend=y[0])                               # +ve = falling

    peak = int(np.argmax(velocity))
    peak_speed = float(velocity[peak])
    if peak_speed <= 0:
        return len(y) - 1, 0.0

    threshold = 0.2 * peak_speed
    after = np.flatnonzero(velocity[peak:] < threshold)
    impact = int(peak + after[0]) if after.size else len(y) - 1

    baseline = float(np.median(np.abs(velocity)))
    confidence = float(np.clip(1.0 - baseline / max(peak_speed, 1e-6), 0.0, 1.0))
    return min(impact, clip.num_frames - 1), confidence


def write_template(clips: list[ClipRecord], out: Path, suggest: bool, annotator: str) -> Path:
    """Write an onsets CSV covering every fall clip in the cache."""
    falls = [c for c in clips if c.is_fall]
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clip_id", "impact_frame", "annotator", "notes",
             "num_frames", "fps", "activity", "view", "suggested", "suggestion_confidence"]
        )
        for clip in sorted(falls, key=lambda c: c.clip_id):
            guess, confidence = suggest_impact_frame(clip)
            writer.writerow([
                clip.clip_id,
                guess if suggest else "",
                annotator if suggest else "",
                "UNREVIEWED heuristic - verify against video" if suggest else "",
                clip.num_frames,
                f"{clip.fps:.2f}",
                clip.activity,
                clip.view,
                guess,
                f"{confidence:.3f}",
            ])

    print(f"wrote {len(falls)} fall clips to {out}")
    if suggest:
        print(
            "  impact_frame is pre-filled with a HEURISTIC and marked UNREVIEWED.\n"
            "  Review every row against the video before training on it."
        )
    else:
        print("  fill in impact_frame per the protocol at the top of this script.")
    return out


def agreement(path_a: Path, path_b: Path, fps: float, tolerance: int) -> dict:
    """Inter-annotator agreement between two completed onset files."""
    a, b = read_onsets(path_a), read_onsets(path_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two files share no annotated clips")

    diff = np.array([a[c] - b[c] for c in common], dtype=np.float64)
    abs_diff = np.abs(diff)

    result = {
        "clips_in_a": len(a),
        "clips_in_b": len(b),
        "clips_compared": len(common),
        "mean_abs_diff_frames": float(abs_diff.mean()),
        "median_abs_diff_frames": float(np.median(abs_diff)),
        "max_abs_diff_frames": float(abs_diff.max()),
        "mean_abs_diff_seconds": float(abs_diff.mean() / fps),
        "mean_signed_diff_frames": float(diff.mean()),
        f"within_{tolerance}_frames": float((abs_diff <= tolerance).mean()),
        "within_1_frame": float((abs_diff <= 1).mean()),
    }

    values_a = np.array([a[c] for c in common], dtype=np.float64)
    values_b = np.array([b[c] for c in common], dtype=np.float64)
    if values_a.std() > 0 and values_b.std() > 0:
        result["pearson_r"] = float(np.corrcoef(values_a, values_b)[0, 1])

    print(f"inter-annotator agreement over {len(common)} clips")
    for key, value in result.items():
        print(f"  {key:28s} {value:.4f}" if isinstance(value, float) else f"  {key:28s} {value}")

    # A systematic offset is worse than random scatter: it shifts every lead
    # time in the same direction rather than averaging out.
    if abs(result["mean_signed_diff_frames"]) > 1.0:
        print(
            f"\n  NOTE: a systematic offset of {result['mean_signed_diff_frames']:+.1f} "
            f"frames between annotators biases every lead time. Reconcile the "
            f"protocol before annotating the rest."
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("PROTOCOL") :],
    )
    parser.add_argument("--cache", help="skeleton cache directory")
    parser.add_argument("--out", default="data/annotations/onsets.csv")
    parser.add_argument("--suggest", action="store_true",
                        help="pre-fill with the kinematic heuristic (needs review)")
    parser.add_argument("--annotator", default="heuristic")
    parser.add_argument("--agreement", nargs=2, metavar=("A.csv", "B.csv"),
                        help="report inter-annotator agreement between two files")
    parser.add_argument("--fps", type=float, default=18.0,
                        help="frame rate used to convert agreement to seconds")
    parser.add_argument("--tolerance", type=int, default=2,
                        help="frames within which two annotators are considered to agree")
    args = parser.parse_args(argv)

    if args.agreement:
        agreement(Path(args.agreement[0]), Path(args.agreement[1]), args.fps, args.tolerance)
        return

    if not args.cache:
        parser.error("--cache is required unless --agreement is given")
    write_template(load_cache(args.cache), Path(args.out), args.suggest, args.annotator)


if __name__ == "__main__":
    main()
