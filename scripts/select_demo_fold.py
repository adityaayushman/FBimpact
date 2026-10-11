"""Choose which fold's checkpoint backs the public demo, by a rule fixed in advance.

    python scripts/select_demo_fold.py

The rule (results/analysis/PLAN.txt, plan section 4.3): take the lowest-index
seed-0 fold of the phase arm whose TEST split contains

  * clips from both UR Fall and Le2i,
  * at least one fall, and
  * at least one Le2i fall whose annotated onset-to-impact is under 15 frames -
    a fast fall, which is the case the phase head is supposed to help with.

It is deliberately blind to how any fold performs. Choosing the demo's fold by how
good it looks would make the demo a flattering sample of the method instead of a
fair one; choosing the lowest qualifying index removes that degree of freedom.

Imports no torch: it reads each run's split.json and the skeleton caches' metadata.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAST_FALL_FRAMES = 15


def clip_meta(cache_dirs: list[Path]) -> dict[str, dict]:
    out = {}
    for cache in cache_dirs:
        for path in sorted(cache.glob("*.npz")):
            with np.load(path, allow_pickle=False) as handle:
                attrs = json.loads(str(handle["attrs"]))
            # Keyed by SUBJECT, because that is what split.json stores. For both
            # datasets the subject is the clip's own group id (a UR Fall sequence,
            # a Le2i video) and differs from the clip id, so keying by clip id
            # silently matches nothing - and "no fold qualifies" would then be a
            # confident-looking answer to a question the script never asked.
            out[attrs["subject"]] = attrs
    return out


def main() -> int:
    meta = clip_meta([ROOT / "data/cache/urfd", ROOT / "data/cache/le2i"])
    index = ROOT / "results/combined/runs.csv"
    runs = [r for r in csv.DictReader(index.open(newline="", encoding="utf-8"))
            if r["config"] == "combined_phases" and int(float(r["seed"])) == 0]
    runs.sort(key=lambda r: int(float(r["fold"])))

    print(f"{'fold':>4} {'test clips':>10} {'urfd':>5} {'le2i':>5} {'falls':>6} {'fast le2i falls':>16}  qualifies")
    chosen = None
    for r in runs:
        run_dir = Path(r["run_dir"])
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        test = set(json.loads((run_dir / "split.json").read_text(encoding="utf-8"))["split"]["test"])
        clips = [meta[s] for s in test if s in meta]
        if len(clips) != len(test):
            # A partial match would make the qualifying test below look at a
            # subset of the fold. Refuse rather than decide on incomplete data.
            raise SystemExit(f"fold {r['fold']}: {len(test) - len(clips)} test subjects "
                             f"not found in the caches")
        sources = {c.get("source") for c in clips}
        falls = [c for c in clips if c.get("label") == "fall"]
        fast = [c for c in falls
                if c.get("source") == "le2i" and c.get("impact_frame") is not None
                and c.get("onset_frame") is not None
                and 0 <= c["impact_frame"] - c["onset_frame"] < FAST_FALL_FRAMES]
        ok = {"urfd", "le2i"} <= sources and len(falls) >= 1 and len(fast) >= 1
        print(f"{int(float(r['fold'])):>4} {len(clips):>10} "
              f"{sum(c.get('source') == 'urfd' for c in clips):>5} "
              f"{sum(c.get('source') == 'le2i' for c in clips):>5} {len(falls):>6} {len(fast):>16}  "
              f"{'YES' if ok else 'no'}")
        if ok and chosen is None:
            chosen = (int(float(r["fold"])), str(run_dir), sorted(test))

    if chosen is None:
        print("\nNo fold qualifies. Per the pre-registration, state that limit; do not pick by appearance.")
        return 1

    fold, run_dir, test = chosen
    print(f"\nSELECTED fold {fold} (lowest qualifying index): {run_dir}")
    out = ROOT / "results/analysis/demo_fold.json"
    out.write_text(json.dumps({
        "rule": "lowest-index seed-0 combined_phases fold whose test split has both datasets, "
                f">=1 fall and >=1 Le2i fall with onset-to-impact < {FAST_FALL_FRAMES} frames",
        "fold": fold, "run_dir": run_dir, "test_clips": test,
        "selected_without_reference_to_performance": True,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
