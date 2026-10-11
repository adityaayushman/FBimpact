"""Paired analysis of the combined-benchmark arms.

    python scripts/analyze_arms.py

Reads the per-clip `outcomes_test.csv` that every run already writes, joins it
with per-clip negative time derived from the skeleton caches, and writes CSVs to
`results/analysis/`. It applies the decision rules pre-registered in
`results/analysis/PLAN.txt`, and it does so mechanically: nothing here is tuned
after looking at a result.

Deliberately imports no torch. The confirmation grid holds the GPU and a large
share of the machine's commit for hours, and this script is meant to be run
alongside it.

Statistical design, and why:

* **The unit is the clip.** A false alarm belongs to a clip and a detection
  belongs to a fall clip. Resampling clips - not frames, not seeds - respects
  the correlation inside a clip and is what a reviewer will ask about.
* **One set of resampled clip indices, reused for every seed and both arms.**
  Seeds are repeated measurements of the same 200 clips, not 4x more clips.
  Pooling them as if independent would shrink the interval by roughly a factor
  of two for no reason except that the same clips were scored again.
* **Paired on identical folds.** Folds depend on `data.split.seed`, not the run
  seed, so every arm and seed scores the same clips and a contrast is a
  difference on the same clips.
* **Seed 0 is shown and never tested.** It generated the hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "combined_xyv": "A control (x,y,vx,vy)",
    "combined": "B + 14-ch features",
    "combined_phases": "C + phase supervision",
    "combined_nocons": "D phases, no consistency",
}

# (name, treatment arm, reference arm, what it isolates)
CONTRASTS = [
    ("B-A", "combined", "combined_xyv", "14-channel features"),
    ("C-B", "combined_phases", "combined", "phase supervision (head + loss + consistency)"),
    ("D-B", "combined_nocons", "combined", "phase head + loss alone"),
    ("C-D", "combined_phases", "combined_nocons", "consistency coupling"),
]


# ----------------------------------------------------------- matched recall (3)
def matched_recall_condition(rows: list[dict], treat: str, ref: str, seeds: list[int]) -> str:
    """Decision-rule condition 3: fewer false alarms at the same recall.

    Returns "met", "not met" or "not computed". The treatment must have a lower
    mean FA/hour than the reference at EVERY recall level that both reach. Levels
    a curve never reaches are skipped, not extrapolated; if no level is common to
    both, the condition cannot be evaluated and is reported as such rather than
    passed by default.
    """
    if not rows:
        return "not computed"

    def mean_at(arm: str, level: float):
        vals = [float(r["fa_per_hour"]) for r in rows
                if r["arm"] == arm and int(r["seed"]) in seeds
                and abs(float(r["recall_level"]) - level) < 1e-9 and r["fa_per_hour"] != ""]
        # A level counts for an arm only if EVERY confirmatory seed reached it;
        # averaging over the seeds that happened to reach it would bias the mean.
        return float(np.mean(vals)) if len(vals) == len(seeds) else None

    levels = sorted({float(r["recall_level"]) for r in rows})
    compared = []
    for level in levels:
        t, r = mean_at(treat, level), mean_at(ref, level)
        if t is not None and r is not None:
            compared.append(t < r)
    if not compared:
        return "not computed"
    return "met" if all(compared) else "not met"


# --------------------------------------------------------------------- loading
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clip_geometry(cache_dirs: list[Path]) -> dict[str, dict]:
    """Per-clip frame count, fps, impact frame and dataset, from the npz caches.

    Read with numpy directly rather than through `data.clips`, which would pull in
    torch via the package `__init__`.
    """
    out: dict[str, dict] = {}
    for cache in cache_dirs:
        for path in sorted(cache.glob("*.npz")):
            with np.load(path, allow_pickle=False) as handle:
                frames = int(handle["keypoints"].shape[0])
                attrs = json.loads(str(handle["attrs"]))
            out[attrs["clip_id"]] = {
                "frames": frames,
                "fps": float(attrs["fps"]),
                "impact": attrs.get("impact_frame"),
                "dataset": attrs.get("source", "unknown"),
                "is_fall": attrs.get("label") == "fall",
            }
    return out


def negative_hours(geom: dict, w_pre: int) -> float:
    """Hours of genuinely-normal footage in one clip.

    Mirrors `evaluation/metrics.py`, which counts frames whose label is 0:
    every frame of an ADL clip, and for a fall only the frames before the
    imminent window. Post-impact frames are masked and contribute nothing.
    """
    if geom["is_fall"] and geom["impact"] is not None:
        frames = max(0, int(geom["impact"]) - w_pre)
    else:
        frames = geom["frames"]
    return frames / geom["fps"] / 3600.0


def load_runs(results_dirs: list[Path], geometry: dict[str, dict]) -> dict:
    """`{(arm, seed): {clip_id: record}}`, one record per clip per arm and seed.

    A run is a fold. Concatenating a seed's five folds gives each of the 200
    clips exactly once; anything else means a fold is missing or duplicated, and
    that is reported rather than analysed.
    """
    runs: dict[tuple[str, int], dict[str, dict]] = {}
    folds: dict[tuple[str, int], set[int]] = {}

    for directory in results_dirs:
        for row in read_csv(directory / "runs.csv"):
            arm, seed = row["config"], int(float(row["seed"]))
            fold = int(float(row.get("fold") or 0))
            run_dir = Path(row["run_dir"])
            run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
            outcomes = read_csv(run_dir / "outcomes_test.csv")
            if not outcomes:
                continue

            cfg_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
            w_pre = next(int(l.split(":")[1]) for l in cfg_text.splitlines()
                         if l.strip().startswith("w_pre:"))

            key = (arm, seed)
            bucket = runs.setdefault(key, {})
            folds.setdefault(key, set()).add(fold)
            for o in outcomes:
                cid = o["clip_id"]
                if cid in bucket:
                    raise SystemExit(f"{arm} seed {seed}: clip {cid} appears in two folds")
                geom = geometry.get(cid)
                if geom is None:
                    raise SystemExit(f"{cid} is in a run but not in the skeleton caches")
                lead = o.get("lead_time") or ""
                bucket[cid] = {
                    "is_fall": o["is_fall"] == "True",
                    "warned": o["warned"] == "True",
                    "lead": float(lead) if lead else None,
                    "fa": int(float(o["false_alarms"])),
                    "neg_h": negative_hours(geom, w_pre),
                    "dataset": geom["dataset"],
                }

    complete = {k: v for k, v in runs.items() if folds[k] == set(range(5))}
    partial = sorted(k for k in runs if k not in complete)
    if partial:
        print("  incomplete (fewer than 5 folds), excluded:",
              ", ".join(f"{a}@s{s}({len(folds[(a, s)])}/5)" for a, s in partial))
    return complete


# ------------------------------------------------------------------- statistics
def summarise(records: dict[str, dict]) -> dict:
    falls = [r for r in records.values() if r["is_fall"]]
    leads = [r["lead"] for r in falls if r["warned"] and r["lead"] is not None]
    neg_h = sum(r["neg_h"] for r in records.values())
    fa = sum(r["fa"] for r in records.values())
    return {
        "falls": len(falls),
        "warned": sum(r["warned"] for r in falls),
        "recall": (sum(r["warned"] for r in falls) / len(falls)) if falls else float("nan"),
        "lead": float(np.mean(leads)) if leads else float("nan"),
        "fa": fa,
        "neg_hours": neg_h,
        "fa_per_hour": fa / neg_h if neg_h else float("nan"),
    }


def ordered_clips(runs: dict, arms: list[str], seeds: list[int]) -> list[str]:
    """Clip ids present in every (arm, seed) requested - the paired universe."""
    sets = [set(runs[(a, s)]) for a in arms for s in seeds if (a, s) in runs]
    return sorted(set.intersection(*sets)) if sets else []


def paired_bootstrap(runs, treat, ref, seeds, universe, n_boot=2000, seed=12345):
    """Seed-averaged paired difference in FA count and recall, clips resampled.

    Strata are (dataset, fall/non-fall): both are fixed properties of a clip, and
    stratifying keeps every resample's mix of UR Fall/Le2i and falls/ADLs equal
    to the real one, so the interval reflects clip sampling and not a lucky
    ratio of easy to hard clips.
    """
    rng = np.random.default_rng(seed)
    probe = runs[(treat, seeds[0])]
    strata: dict[tuple, list[int]] = {}
    for i, cid in enumerate(universe):
        strata.setdefault((probe[cid]["dataset"], probe[cid]["is_fall"]), []).append(i)

    def stack(arm, field):
        return np.array([[runs[(arm, s)][cid][field] for cid in universe] for s in seeds], dtype=float)

    t_fa, r_fa = stack(treat, "fa"), stack(ref, "fa")
    t_hit, r_hit = stack(treat, "warned"), stack(ref, "warned")
    is_fall = np.array([runs[(treat, seeds[0])][cid]["is_fall"] for cid in universe])

    d_fa = np.empty(n_boot)
    d_rec = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata.values()])
        # Same indices for every seed and both arms, then average over seeds.
        d_fa[b] = (t_fa[:, idx].sum(axis=1) - r_fa[:, idx].sum(axis=1)).mean()
        f = idx[is_fall[idx]]
        # A clip set with no falls has no recall to compare; say so instead of
        # letting NumPy average an empty slice into a warning and a NaN.
        d_rec[b] = (((t_hit[:, f].mean(axis=1)) - (r_hit[:, f].mean(axis=1))).mean()
                    if len(f) else np.nan)

    lo_fa, hi_fa = np.percentile(d_fa, [2.5, 97.5])
    lo_r, hi_r = (np.nanpercentile(d_rec, [2.5, 97.5]) if np.isfinite(d_rec).any()
                  else (np.nan, np.nan))
    return {
        "d_fa": float((t_fa.sum(axis=1) - r_fa.sum(axis=1)).mean()),
        "d_fa_lo": float(lo_fa), "d_fa_hi": float(hi_fa),
        "d_recall": (float(np.mean([
            t_hit[k][is_fall].mean() - r_hit[k][is_fall].mean() for k in range(len(seeds))
        ])) if is_fall.any() else float("nan")),
        "d_recall_lo": float(lo_r), "d_recall_hi": float(hi_r),
    }


# ------------------------------------------------------------------------- main
def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", nargs="*", default=["results/combined", "results/combined_confirm"])
    parser.add_argument("--caches", nargs="*", default=["data/cache/urfd", "data/cache/le2i"])
    parser.add_argument("--confirmatory", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--descriptive", nargs="*", type=int, default=[0])
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args(argv)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    geometry = clip_geometry([ROOT / c for c in args.caches])
    print(f"{len(geometry)} clips in caches")
    runs = load_runs([ROOT / r for r in args.results], geometry)
    print(f"{len(runs)} complete (arm, seed) cells: "
          + ", ".join(f"{a}@s{s}" for a, s in sorted(runs)))

    # 1. per arm and seed ------------------------------------------------------
    rows = []
    for (arm, seed), recs in sorted(runs.items()):
        s = summarise(recs)
        rows.append({"arm": arm, "seed": seed,
                     "role": "confirmatory" if seed in args.confirmatory else "descriptive",
                     **{k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()}})
    write_csv(out / "arms_by_seed.csv", rows)

    print("\nper arm and seed (seed 0 descriptive; confirmatory seeds are what counts)")
    print(f"{'arm':<17}{'seed':>5}{'role':>14}{'recall':>9}{'warned':>8}{'lead':>8}{'FA':>6}{'FA/h':>9}")
    for r in rows:
        print(f"{r['arm']:<17}{r['seed']:>5}{r['role']:>14}{fmt(r['recall']):>9}"
              f"{r['warned']:>5}/{r['falls']:<2}{fmt(r['lead']):>8}{r['fa']:>6}{fmt(r['fa_per_hour'], 1):>9}")

    # 2. across confirmatory seeds --------------------------------------------
    summary = []
    for arm in ARMS:
        cells = [r for r in rows if r["arm"] == arm and r["seed"] in args.confirmatory]
        if not cells:
            continue
        rec = np.array([c["recall"] for c in cells])
        fa = np.array([c["fa"] for c in cells], dtype=float)
        lead = np.array([c["lead"] for c in cells])
        summary.append({"arm": arm, "n_seeds": len(cells),
                        "recall_mean": rec.mean(), "recall_sd": rec.std(ddof=1) if len(cells) > 1 else float("nan"),
                        "fa_mean": fa.mean(), "fa_sd": fa.std(ddof=1) if len(cells) > 1 else float("nan"),
                        "lead_mean": np.nanmean(lead)})
    write_csv(out / "arms_summary.csv", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in summary])

    # 3. contrasts -------------------------------------------------------------
    matched_rows = read_csv(out / "matched_recall.csv")
    contrast_rows = []
    print("\ncontrasts (confirmatory seeds only; a negative d_FA means fewer false alarms)")
    if not matched_rows:
        print("  (no matched_recall.csv yet: condition 3 will read 'not computed')")
    for name, treat, ref, meaning in CONTRASTS:
        seeds = [s for s in args.confirmatory if (treat, s) in runs and (ref, s) in runs]
        if not seeds:
            print(f"  {name}: no complete confirmatory seeds yet")
            continue
        universe = ordered_clips(runs, [treat, ref], seeds)
        stats = paired_bootstrap(runs, treat, ref, seeds, universe, n_boot=args.boot)
        per_seed = [summarise(runs[(treat, s)])["fa"] - summarise(runs[(ref, s)])["fa"] for s in seeds]
        fewer = sum(1 for d in per_seed if d < 0)
        cond1 = stats["d_fa_hi"] < 0
        cond2 = fewer >= (3 if len(seeds) == 4 else max(1, int(np.ceil(0.75 * len(seeds)))))
        rec_ok = stats["d_recall_lo"] > 0
        cond3 = matched_recall_condition(matched_rows, treat, ref, seeds)
        if not (cond1 and cond2):
            verdict = "not shown"
        elif cond3 == "met":
            verdict = "reduces false alarms (all three conditions met)"
        elif cond3 == "not met":
            verdict = "fewer false alarms at the default threshold; not shown at matched recall"
        else:
            verdict = "conditions 1-2 met; condition 3 (matched recall) not computed"
        contrast_rows.append({
            "contrast": name, "means": meaning, "seeds": len(seeds), "clips": len(universe),
            **{k: round(v, 4) for k, v in stats.items()},
            "seeds_fewer_fa": f"{fewer}/{len(seeds)}",
            "cond1_ci_below_zero": cond1, "cond2_seed_majority": cond2,
            "cond3_matched_recall": cond3,
            "recall_ci_excludes_zero": rec_ok,
            "fa_verdict": verdict,
        })
        print(f"  {name:<4} dFA {stats['d_fa']:+6.1f} [{stats['d_fa_lo']:+6.1f}, {stats['d_fa_hi']:+6.1f}]  "
              f"dRecall {stats['d_recall']:+.3f} [{stats['d_recall_lo']:+.3f}, {stats['d_recall_hi']:+.3f}]  "
              f"fewer FA in {fewer}/{len(seeds)} seeds  -> {contrast_rows[-1]['fa_verdict']}")
    write_csv(out / "contrasts.csv", contrast_rows)

    # 4. per dataset -----------------------------------------------------------
    ds_rows = []
    for (arm, seed), recs in sorted(runs.items()):
        for dataset in sorted({r["dataset"] for r in recs.values()}):
            sub = {c: r for c, r in recs.items() if r["dataset"] == dataset}
            s = summarise(sub)
            ds_rows.append({"arm": arm, "seed": seed, "dataset": dataset,
                            **{k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()}})
    write_csv(out / "arms_by_dataset.csv", ds_rows)

    print(f"\nwrote {out}/arms_by_seed.csv, arms_summary.csv, contrasts.csv, arms_by_dataset.csv")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
