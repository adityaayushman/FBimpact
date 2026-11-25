"""Figures for the IEEE paper, drawn from the run data (nothing typed in by hand).

    python paper/make_figures.py

Writes vector PDF (for LaTeX) and 300 dpi PNG (for the PDF preview) to paper/figures/.
Single-column figures are 3.5 in wide and double-column figures 7.16 in, as IEEE requires.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, ACCENT, GREY = "#1a1a1a", "#7a1f3d", "#8a8a8a"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 7.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------- Fig. 1
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, shrinkA=0, shrinkB=0))

    y, h = 17, 9
    xs = [1.5, 16.5, 34, 51, 68.5, 85]
    ws = [12.5, 14.5, 14.5, 14.5, 14, 13.5]
    labels = ["RGB video", "Pose estimator\n(frozen, COCO-17)", "Normalise,\n14-channel features", "25-frame window\non body graph", "Causal ST-GCN\n(per-frame logit)", "Threshold $\\tau$,\npersistence $k$"]
    for x, w, t in zip(xs, ws, labels):
        box(x, y, w, h, t)
    for i in range(5):
        arrow(xs[i] + ws[i] + 0.3, y + h / 2, xs[i + 1] - 0.3, y + h / 2)
    ax.plot([32.5, 32.5], [14.5, 29.0], ls=(0, (3, 2)), lw=0.9, color=ACCENT)
    ax.text(32.5, 29.5, "privacy boundary: only skeletons beyond this line", ha="center", va="bottom", fontsize=6.8, color=ACCENT, style="italic")

    box(83, 2.5, 16, 8.5, "Warning at $t_{warn}$;\n$L=(t^*-t_{warn})/fps$", fc="#f3e8ed", ec=ACCENT)
    arrow(91.75, y - 0.3, 91.0, 11.3)
    box(48, 2.5, 30, 8.5, "Joint relevance at the warning frame;\ndeletion / insertion vs random order")
    arrow(75.5, y - 0.3, 66, 11.3)
    ax.text(72.6, 13.4, "$p_t$", fontsize=7, ha="left", va="center")
    arrow(78.3, 6.75, 82.7, 6.75)
    ax.text(1.5, 7.5, "A. Pose (frozen)   B. Features   C. Window and graph\nD. Model   E-F. Evidence   G-H. Decision and lead time", fontsize=6.8, va="center", color="#444444")
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------------------------------- Fig. 2
def fig_urfall() -> None:
    data = rows(ROOT / "results" / "urfd" / "runs.csv")
    names = [("no_preimpact_loss", "No pre-impact loss"), ("no_grounding", "No grounding head"), ("baseline_stgcn", "Baseline ST-GCN"),
             ("ours_preimpact", "Full model ($\\lambda$=1.5)"), ("no_velocity", "No velocity features"), ("no_temporal", "No temporal modelling")]
    per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in data:
        cell = per_seed[r["config"]][int(float(r["seed"]))]
        cell[0] += float(r["num_warned"])
        cell[1] += float(r["num_falls"])
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for i, (key, label) in enumerate(reversed(names)):
        vals = [w / f for w, f in per_seed[key].values()]
        pooled = sum(w for w, _ in per_seed[key].values()) / sum(f for _, f in per_seed[key].values())
        ax.scatter(vals, [i] * len(vals), s=14, color=GREY, zorder=2, label="per seed" if i == 0 else None)
        ax.scatter([pooled], [i], marker="D", s=26, color=ACCENT if key == "ours_preimpact" else INK, zorder=3, label="pooled" if i == 0 else None)
        ax.text(pooled + 0.03, i + 0.22, f"{pooled:.3f}", fontsize=6.8, va="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n for _, n in reversed(names)])
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Recall on UR Fall (30 falls)")
    ax.grid(axis="x", lw=0.3, color="#cccccc")
    ax.legend(loc="lower right", handletextpad=0.2, borderpad=0.1)
    save(fig, "fig2_urfall")


# ---------------------------------------------------------------------------------------- Fig. 3
def fig_confirmation() -> None:
    by_seed = rows(ROOT / "results" / "analysis" / "arms_by_seed.csv")
    fa = {(r["arm"], int(r["seed"])): float(r["fa"]) for r in by_seed}
    matched = rows(ROOT / "results" / "analysis" / "matched_recall.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.34})
    seeds = [1, 2, 3, 4]
    for arm, label, style, color in (("combined", "B (14-channel features)", "o-", INK), ("combined_phases", "C (+ four-phase supervision)", "s--", ACCENT)):
        a.plot(seeds, [fa[(arm, s)] for s in seeds], style, ms=4.5, lw=0.9, color=color, label=label, mfc="white" if arm != "combined" else color)
    a.set_xticks(seeds)
    a.set_xlabel("Training seed")
    a.set_ylabel("False alarms (count, 129 falls)")
    a.set_ylim(60, 150)
    a.grid(lw=0.3, color="#cccccc")
    a.legend(loc="upper right")
    a.set_title("(a) Per-seed false alarms, arm B versus arm C", fontsize=8)
    arms = [("combined_xyv", "A"), ("combined", "B"), ("combined_phases", "C"), ("combined_nocons", "D")]
    for i, (arm, tag) in enumerate(arms):
        for lvl, off, mk in (("0.6", -0.17, "o"), ("0.65", 0.17, "s")):
            v = [float(r["fa_per_hour"]) for r in matched if r["arm"] == arm and r["recall_level"] == lvl and r["seed"] in ("1", "2", "3", "4") and r["fa_per_hour"]]
            m = sum(v) / len(v)
            b.errorbar([i + off], [m], yerr=[[m - min(v)], [max(v) - m]], fmt=mk, ms=4.5, color=INK if lvl == "0.6" else ACCENT, mfc=INK if lvl == "0.6" else "white", capsize=2, lw=0.8,
                       label=f"recall {float(lvl):.2f}" if i == 0 else None)
    b.set_xticks(range(4))
    b.set_xticklabels([t for _, t in arms])
    b.set_xlabel("Arm")
    b.set_ylabel("False alarms per hour at fixed recall")
    b.set_ylim(0, 135)
    b.grid(axis="y", lw=0.3, color="#cccccc")
    b.legend(loc="upper right")
    b.set_title("(b) Matched-recall operating points (mean, range)", fontsize=8)
    save(fig, "fig3_confirmation")


# ---------------------------------------------------------------------------------------- Fig. 4
def fig_heldout() -> None:
    x = rows(ROOT / "results" / "analysis" / "xdomain_by_run.csv")
    confirm = rows(ROOT / "results" / "combined_confirm" / "runs.csv")
    ref = {}
    for arm, cfg in (("ctrl", "combined_xyv"), ("phases", "combined_phases")):
        cell = [r for r in confirm if r["config"] == cfg and int(float(r["seed"])) in (1, 2, 3)]
        ref[arm] = sum(float(r["num_warned"]) for r in cell) / sum(float(r["num_falls"]) for r in cell)
    protos = [("urfd_to_le2i", "UR Fall $\\rightarrow$ Le2i\n(99 falls)"), ("le2i_to_urfd", "Le2i $\\rightarrow$ UR Fall\n(30 falls)"),
              ("leave_coffee_room_out", "Leave Coffee room out\n(62 falls)"), ("leave_home_out", "Leave Home out\n(37 falls)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (key, label) in enumerate(reversed(protos)):
        for arm, dy, mk, mfc in (("ctrl", -0.13, "o", INK), ("phases", 0.13, "o", "white")):
            v = [float(r["recall"]) for r in x if r["protocol"] == key and r["arm"] == arm]
            ax.scatter(v, [i + dy] * len(v), s=17, marker=mk, facecolors=mfc, edgecolors=INK if arm == "ctrl" else ACCENT, linewidths=1.0, zorder=3,
                       label=("control" if arm == "ctrl" else "four-phase") if i == 0 else None)
    ax.axvline(ref["ctrl"], color=ACCENT, ls=(0, (4, 2)), lw=0.9, zorder=1)
    ax.text(ref["ctrl"] + 0.01, 3.55, f"in-domain {ref['ctrl']:.2f}", fontsize=6.8, color=ACCENT, va="center")
    ax.set_yticks(range(len(protos)))
    ax.set_yticklabels([lbl for _, lbl in reversed(protos)])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlabel("Recall on the held-out domain (3 seeds)")
    ax.grid(axis="x", lw=0.3, color="#cccccc")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, borderpad=0.1, ncol=2, columnspacing=1.2)
    save(fig, "fig4_heldout")


# ---------------------------------------------------------------------------------------- Fig. 5
def fig_faithfulness() -> None:
    d = json.loads((ROOT / "frontend" / "lib" / "faithfulness.json").read_text(encoding="utf-8"))
    rows_ = [r for r in d["rows"] if r["benchmark"] != "Synthetic fixture"]
    pick = [("UR Fall + Le2i (combined)", "combined_phases", "Combined, C (four-phase)"), ("UR Fall + Le2i (combined)", "combined", "Combined, B (14-channel)"),
            ("Le2i", "ours_preimpact", "Le2i, full model"), ("Le2i", "baseline_stgcn", "Le2i, baseline"),
            ("UR Fall", "ours_preimpact", "UR Fall, full model"), ("UR Fall", "baseline_stgcn", "UR Fall, baseline")]
    entries = [("All real-data runs (90)", d["overall"]["deletion"], d["overall"]["insertion"])]
    for bench, variant, label in pick:
        r = next(x for x in rows_ if x["benchmark"] == bench and x["variant"] == variant and x["method"] == "attention")
        entries.append((f"{label} ({r['runs']})", r["deletion"], r["insertion"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, key, title in ((axes[0], 0, "(a) Deletion gap"), (axes[1], 1, "(b) Insertion gap")):
        for i, (label, dele, ins) in enumerate(reversed(entries)):
            p = dele if key == 0 else ins
            top = (i == len(entries) - 1)
            col = ACCENT if top else INK
            ax.plot([p["lo"], p["hi"]], [i, i], color=col, lw=1.2 if top else 0.9)
            ax.plot([p["mean"]], [i], "D" if top else "o", ms=4.8 if top else 4, color=col)
        ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(-0.27, 0.23)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Gap above random joint order (95% CI over runs)")
        ax.grid(axis="x", lw=0.3, color="#cccccc")
    axes[0].set_yticks(range(len(entries)))
    axes[0].set_yticklabels([e[0] for e in reversed(entries)])
    save(fig, "fig5_faithfulness")


# ---------------------------------------------------------------------------------------- Architecture
def fig_architecture() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from data.skeleton import BONES

    fig = plt.figure(figsize=(7.16, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 43)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=INK, fs=6.6, ls="-", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.7", fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=INK, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, shrinkA=0, shrinkB=0, ls=ls))

    ax.text(1, 41.2, "(a) Network", fontsize=8, fontweight="bold", va="center")
    box(1, 17, 9.5, 13, "Input\n$C\\times T\\times V$\n14 $\\times$ 25 $\\times$ 17")
    box(13, 20.5, 7, 6, "Batch-\nNorm")
    arrow(10.8, 23.5, 12.7, 23.5)
    arrow(20.3, 23.5, 22.4, 23.5)
    ax.add_patch(FancyBboxPatch((22.5, 5), 20.5, 33, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#f7f7f7", ec=GREY, lw=0.8, ls=(0, (3, 2))))
    ax.text(32.75, 40, "ST-GCN block, 4 times\n(64, 64, 128, 128 channels)", ha="center", va="center", fontsize=6.6, color="#333333")
    box(24, 29.5, 17.5, 6.5, "Spatial graph conv, 3 partitions\n$\\sum_k (A_k\\odot M_k)\\,X\\,W_k$", fs=6.2)
    box(24, 22.5, 17.5, 5, "BN, ReLU", fs=6.2)
    box(24, 14.5, 17.5, 6.5, "Causal temporal conv\nkernel 7, left-padded", fs=6.2)
    box(24, 8.5, 17.5, 4.3, "BN, dropout, + residual, ReLU", fs=6.0)
    for y0, y1 in ((29.3, 27.7), (22.3, 21.2), (14.3, 12.9)):
        arrow(32.75, y0, 32.75, y1)
    arrow(43.2, 23.5, 45.4, 23.5)
    box(45.5, 17, 13, 13, "Joint attention\n$a_{t,v}=\\mathrm{softmax}_v(\\cdot)$\n$z_t=\\sum_v a_{t,v}h_{t,v}$", fs=6.2)
    arrow(58.7, 25.5, 60.5, 29.5)
    arrow(58.7, 21.5, 60.5, 17.5)
    box(60.5, 27, 7.2, 7, "1$\\times$1 conv\nlogit,\n$p_t$", fs=6.2)
    box(60.5, 11.5, 7.2, 7, "Phase\nhead (4)\noptional", fs=6.0, ls=(0, (3, 2)))
    ax.text(1, 11.6, "One score per frame,\nno pooling over time", fontsize=6.2, color="#444444", style="italic", linespacing=1.2)

    ax.plot([69.2, 69.2], [2, 41], color="#cccccc", lw=0.7)
    ax.text(70.5, 41.2, "(b) Body graph", fontsize=8, fontweight="bold", va="center")
    pos = {0: (0, 9.0), 1: (-0.32, 9.35), 2: (0.32, 9.35), 3: (-0.7, 9.0), 4: (0.7, 9.0), 5: (-1.25, 7.6), 6: (1.25, 7.6), 7: (-1.7, 5.9), 8: (1.7, 5.9),
           9: (-1.8, 4.2), 10: (1.8, 4.2), 11: (-0.8, 4.7), 12: (0.8, 4.7), 13: (-0.95, 2.4), 14: (0.95, 2.4), 15: (-1.0, 0.2), 16: (1.0, 0.2)}
    sx, sy, ox, oy = 3.2, 2.85, 77.2, 9.5
    P = {j: (ox + x * sx, oy + y * sy) for j, (x, y) in pos.items()}
    for a_, b_ in BONES:
        ax.plot([P[a_][0], P[b_][0]], [P[a_][1], P[b_][1]], color=GREY, lw=1.0, zorder=1)
    for j, (x, y) in P.items():
        centre = j in (11, 12)
        ax.scatter([x], [y], s=16 if centre else 11, color=ACCENT if centre else INK, zorder=3)
    ax.text(ox, oy + 3.55 * sy, "centre\n(hips)", fontsize=5.8, color=ACCENT, va="center", ha="center", linespacing=1.1)
    ax.text(77.2, 6.6, "17 COCO joints; each joint's\nneighbourhood is split into itself,\nneighbours nearer the hips and\nneighbours farther from them", fontsize=5.9, ha="center", va="top", color="#333333", linespacing=1.15)

    ax.plot([86.5, 86.5], [2, 41], color="#cccccc", lw=0.7)
    ax.text(87.4, 41.2, "(c) Causal receptive field", fontsize=7.3, fontweight="bold", va="center")
    spans = [25, 19, 13, 7, 1]
    labels = ["input", "after block 1", "after block 2", "after block 3", "output $p_t$"]
    x0, x1_, unit = 88.0, 98.6, (98.6 - 88.0) / 25
    for i, (n, lab) in enumerate(zip(spans, labels)):
        y = 6.0 + i * 6.2
        left = x1_ - n * unit
        ax.add_patch(FancyBboxPatch((left, y), n * unit, 3.2, boxstyle="square,pad=0", fc="#f3e8ed" if i == 4 else "#e8e8e8", ec=INK, lw=0.7))
        ax.text(x0 - 0.3, y + 1.6, lab, fontsize=5.6, ha="right", va="center") if False else None
        ax.text(98.9, y + 1.6, lab, fontsize=5.6, ha="left", va="center") if False else None
    for i, lab in enumerate(labels):
        ax.text(88.0, 7.6 + i * 6.2 + 3.1, lab, fontsize=5.6, ha="left", va="bottom")
    ax.annotate("", xy=(98.6 - 0.5 * unit, 6.0 + 4 * 6.2), xytext=(98.6 - 0.5 * unit, 9.3), arrowprops=dict(arrowstyle="-", lw=0.5, color=ACCENT, ls=(0, (2, 2))))
    ax.text(93.3, 3.3, "past  $\\leftarrow$  frames  $\\rightarrow$  $t$", fontsize=5.8, ha="center", va="center", color="#333333")
    ax.text(93.3, 0.9, "RF = 1 + 4(7 - 1) = 25", fontsize=5.8, ha="center", va="center", color=ACCENT)
    save(fig, "fig2_architecture")


# ---------------------------------------------------------------------------------------- Protocol
def fig_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=INK, fs=6.5, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.7", fc=fc, ec=ec, lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, shrinkA=0, shrinkB=0))

    ax.text(0.5, 36.6, "Decision rules and analysis plan committed before the first run they govern", fontsize=6.8, color=ACCENT, style="italic", va="center")
    ax.plot([0.5, 99.5], [34.6, 34.6], color=ACCENT, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.5, 31.9, "In-domain (confirmation grid)", fontsize=7, fontweight="bold", va="center")
    y, h = 20.5, 8
    xs = [0.5, 17.5, 34.5, 51.5, 68.5, 85]
    ws = [15, 15, 15, 15, 15, 14.5]
    txt = ["200 clips\n(UR Fall + Le2i)", "5 folds, fixed split\nseed (by clip)", "Train arm A to D,\nseeds 1 to 4\n(early stopping)", "Choose $(\\tau, k)$ on\nvalidation clips,\nfreeze", "Streaming test\non held-out fold\n(Algorithm 1)", "Pool folds per seed;\npaired bootstrap,\nrule (Alg. 4)"]
    for x, w, t in zip(xs, ws, txt):
        box(x, y, w, h, t)
    for i in range(5):
        arrow(xs[i] + ws[i] + 0.2, y + h / 2, xs[i + 1] - 0.2, y + h / 2)
    ax.text(0.5, 14.6, "Held-out domain (generalisation)", fontsize=7, fontweight="bold", va="center")
    y2 = 4.5
    xs2 = [0.5, 25.5, 50.5, 75.5]
    ws2 = [22, 22, 22, 24]
    txt2 = ["Train on one dataset or\nscene set; validation clips\nfrom the training side only", "Train control arm and\nfour-phase arm,\nseeds 1 to 3", "Choose $(\\tau, k)$ on\nvalidation clips,\nfreeze", "Test on the unseen room or\ndataset (24 runs); verify no\nclip leaves the test domain"]
    for x, w, t in zip(xs2, ws2, txt2):
        box(x, y2, w, 8, t, fc="#f7f7f7")
    for i in range(3):
        arrow(xs2[i] + ws2[i] + 0.2, y2 + 4, xs2[i + 1] - 0.2, y2 + 4)
    save(fig, "fig3_protocol")


if __name__ == "__main__":
    fig_architecture()
    fig_protocol()
    fig_pipeline()
    fig_urfall()
    fig_confirmation()
    fig_heldout()
    fig_faithfulness()
