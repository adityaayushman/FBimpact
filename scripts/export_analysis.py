"""Turn the confirmatory analysis CSVs into the JSON the website renders.

    python scripts/export_analysis.py

Reads what `scripts/analyze_arms.py` wrote to `results/analysis/` and emits
`frontend/lib/analysis.json`. If the analysis has not been run yet the file says
so (`status: "pending"`) and the site shows a pending state - it never shows a
placeholder that could be read as a result.

The site wording for each outcome is chosen HERE, from the verdict the analysis
computed, using the table in the pre-registered plan. Keeping the mapping in one
place means no component can quietly promote "not shown" into "improves".
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"

# Verdict (from analyze_arms.py) -> what the site may say. Mirrors the claim
# table in the plan; anything not listed falls through to the most conservative.
WORDING = {
    "reduces false alarms (all three conditions met)": (
        "supported",
        "Reduced false alarms in the confirmation seeds, at the default threshold and at matched recall.",
    ),
    "fewer false alarms at the default threshold; not shown at matched recall": (
        "partial",
        "Fewer false alarms at the default threshold; not shown at matched recall, so this may be "
        "a threshold shift rather than a real reduction.",
    ),
    "conditions 1-2 met; condition 3 (matched recall) not computed": (
        "pending",
        "Fewer false alarms at the default threshold in the confirmation seeds; the matched-recall "
        "check has not been run, so no general claim is made yet.",
    ),
    "not shown": (
        "not_shown",
        "Not reproducibly different from seed-to-seed variation in the confirmation seeds.",
    ),
}
FALLBACK = ("not_shown", "Not shown.")


def read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def main() -> int:
    contrasts = []
    for row in read(ANALYSIS / "contrasts.csv"):
        level, text = WORDING.get(row.get("fa_verdict", ""), FALLBACK)
        contrasts.append({
            "contrast": row["contrast"], "means": row["means"],
            "seeds": int(float(row["seeds"])), "clips": int(float(row["clips"])),
            "d_fa": num(row["d_fa"]), "d_fa_lo": num(row["d_fa_lo"]), "d_fa_hi": num(row["d_fa_hi"]),
            "d_recall": num(row["d_recall"]), "d_recall_lo": num(row["d_recall_lo"]),
            "d_recall_hi": num(row["d_recall_hi"]),
            "seeds_fewer_fa": row["seeds_fewer_fa"],
            "matched_recall": row["cond3_matched_recall"],
            "verdict": row["fa_verdict"], "level": level, "wording": text,
        })

    arms = [{
        "arm": r["arm"], "n_seeds": int(float(r["n_seeds"])),
        "recall_mean": num(r["recall_mean"]), "recall_sd": num(r["recall_sd"]),
        "fa_mean": num(r["fa_mean"]), "fa_sd": num(r["fa_sd"]), "lead_mean": num(r["lead_mean"]),
    } for r in read(ANALYSIS / "arms_summary.csv")]

    by_seed = [{
        "arm": r["arm"], "seed": int(float(r["seed"])), "role": r["role"],
        "recall": num(r["recall"]), "warned": int(float(r["warned"])), "falls": int(float(r["falls"])),
        "fa": int(float(r["fa"])), "lead": num(r["lead"]),
    } for r in read(ANALYSIS / "arms_by_seed.csv")]

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete" if contrasts else "pending",
        "plan": "results/analysis/PLAN.txt",
        "contrasts": contrasts, "arms": arms, "by_seed": by_seed,
    }
    out = ROOT / "frontend" / "lib" / "analysis.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: status={payload['status']}, {len(contrasts)} contrasts, {len(by_seed)} seed rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
