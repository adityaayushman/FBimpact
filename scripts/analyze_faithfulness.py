"""RQ2: are the joint explanations faithful, or only plausible-looking?

    python scripts/analyze_faithfulness.py

Pools every `faithfulness_test.json` under results/ (one per trained run) into one answer
per (benchmark, variant): is the model's joint ranking better than a random ordering of the
same joints on the same model?

  deletion gap  = AUC(score) after deleting joints in RANDOM order  -  AUC when deleting in
                  RELEVANCE order. Positive: removing the joints the explanation names hurts the
                  score more than removing random joints.
  insertion gap = AUC when inserting joints in RELEVANCE order  -  RANDOM order. Positive:
                  adding the named joints back restores the score faster.

The unit of analysis is the trained run, not the explanation: explanations from one model are
not independent, so each run contributes one gap, weighted by how many explanations it scored.
Intervals come from resampling runs (seeded). Output: results/analysis/faithfulness_summary.csv
and frontend/lib/faithfulness.json. Nothing here needs torch.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"urfd": "UR Fall", "le2i": "Le2i", "combined": "UR Fall + Le2i (combined)", "bench": "Synthetic fixture"}
BOOT = 10_000


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def pooled(values: list[float], weights: list[float], seed: int = 0) -> dict:
    """Weighted mean over runs with a bootstrap 95% interval (runs resampled)."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    est = weighted_mean(v, w)
    if len(v) < 2:
        return {"mean": est, "lo": None, "hi": None, "runs": len(v), "share_positive": float(np.mean(v > 0))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    boots = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": est, "lo": float(lo), "hi": float(hi), "runs": len(v), "share_positive": float(np.mean(v > 0))}


def collect(results_root: Path) -> list[dict]:
    rows = []
    for path in sorted(results_root.glob("*/*/faithfulness_test.json")):
        source = path.parent.parent.name
        if source not in SOURCES:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        n = int(d.get("num_explanations") or 0)
        if n <= 0 or d.get("deletion_gap") is None:
            continue
        variant = path.parent.name.rsplit("_seed", 1)[0]
        rows.append({
            "source": source, "variant": variant, "method": d.get("method", "?"), "baseline": d.get("baseline", "?"),
            "n": n, "deletion_gap": float(d["deletion_gap"]), "insertion_gap": float(d["insertion_gap"]),
            "run": path.parent.name,
        })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source"], r["variant"], r["method"], r["baseline"]), []).append(r)
    out = []
    for (source, variant, method, baseline), g in sorted(groups.items()):
        w = [r["n"] for r in g]
        out.append({
            "benchmark": SOURCES[source], "variant": variant, "method": method, "baseline": baseline,
            "runs": len(g), "explanations": int(sum(w)),
            "deletion": pooled([r["deletion_gap"] for r in g], w),
            "insertion": pooled([r["insertion_gap"] for r in g], w),
        })
    return out


def main() -> int:
    rows = collect(ROOT / "results")
    if not rows:
        print("no faithfulness results found")
        return 1
    table = summarise(rows)

    allw = [r["n"] for r in rows]
    overall = {
        "runs": len(rows), "explanations": int(sum(allw)),
        "deletion": pooled([r["deletion_gap"] for r in rows], allw),
        "insertion": pooled([r["insertion_gap"] for r in rows], allw),
    }

    out_csv = ROOT / "results" / "analysis" / "faithfulness_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "variant", "method", "baseline", "runs", "explanations",
                    "deletion_gap", "del_lo", "del_hi", "del_share_positive",
                    "insertion_gap", "ins_lo", "ins_hi", "ins_share_positive"])
        for r in table:
            d, i = r["deletion"], r["insertion"]
            w.writerow([r["benchmark"], r["variant"], r["method"], r["baseline"], r["runs"], r["explanations"],
                        round(d["mean"], 4), d["lo"] and round(d["lo"], 4), d["hi"] and round(d["hi"], 4), round(d["share_positive"], 3),
                        round(i["mean"], 4), i["lo"] and round(i["lo"], 4), i["hi"] and round(i["hi"], 4), round(i["share_positive"], 3)])

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "overall": overall, "rows": table, "bootstrap_reps": BOOT}
    (ROOT / "frontend" / "lib" / "faithfulness.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    def fmt(p):
        return f"{p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]" if p["lo"] is not None else f"{p['mean']:+.3f}"
    print(f"{len(rows)} runs, {overall['explanations']} explanations\n")
    print(f"OVERALL  deletion gap {fmt(overall['deletion'])}   insertion gap {fmt(overall['insertion'])}\n")
    print(f"{'benchmark':27s}{'variant':20s}{'runs':>5s}{'expl':>6s}   {'deletion gap [95% CI]':26s}{'insertion gap [95% CI]':26s}")
    for r in table:
        print(f"{r['benchmark'][:26]:27s}{r['variant'][:19]:20s}{r['runs']:5d}{r['explanations']:6d}   {fmt(r['deletion']):26s}{fmt(r['insertion']):26s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
