"""Regenerate the paper/site figures from committed CSVs.

    python docs/figures/make_figures.py

fig_retrieval.(png|pdf)  R@1 by method across three levels of ticket clarity
                         data: brain/benchmark/repro/results/tiered_eval.csv
fig_pipeline.(png|pdf)   system diagram (no data)
fig_envelope.(png|pdf)   required rest-to-contact joint offsets vs the action envelope
                         data: docs/figures/data/action_envelope.csv

Style follows the lab paper guide: text >= 20 pt, lines >= 3 px, every series distinguishable
in black and white (hatching), color used only to separate the learned model from baselines
and learned modules from rule-based ones.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "brain" / "benchmark" / "repro" / "results"

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"
BLUE_STRONG, BLUE_MID, BLUE_LIGHT = "#1c5cab", "#5598e7", "#9ec5f4"
GREY_DARK, GREY_LIGHT = "#7d7c76", "#c9c8c2"

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 22, "xtick.labelsize": 20,
    "ytick.labelsize": 20, "legend.fontsize": 20, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.linewidth": 1.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "hatch.linewidth": 1.5,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / f"{name}.png")


def fig_retrieval():
    df = pd.read_csv(RESULTS / "tiered_eval.csv")
    df["family"] = df.method.where(~df.method.str.startswith("ft_minilm"), "ft_minilm")
    agg = df.groupby(["tier", "family"])["R@1"].agg(["mean", "std", "count"]).reset_index()
    tiers = [("hard", "Test tickets"), ("machine_named", "Test tickets,\nmachine named"),
             ("informed", "Tickets written\nfrom SOP titles")]
    methods = [  # (key, label, color, hatch)
        ("tfidf", "TF-IDF", GREY_LIGHT, "//"),
        ("bm25", "BM25", GREY_DARK, "\\\\"),
        ("zs_minilm", "MiniLM, untuned", BLUE_LIGHT, ".."),
        ("zs_bge_small", "bge-small, untuned", BLUE_MID, "xx"),
        ("ft_minilm", "MiniLM, fine-tuned (ours)", BLUE_STRONG, ""),
    ]
    n_ft = int(agg[agg.family == "ft_minilm"]["count"].max())
    fig, ax = plt.subplots(figsize=(15, 7.5))
    w, gap = 0.16, 0.005
    x = np.arange(len(tiers))
    for i, (key, label, color, hatch) in enumerate(methods):
        rows = [agg[(agg.tier == t) & (agg.family == key)].iloc[0] for t, _ in tiers]
        y = np.array([r["mean"] for r in rows])
        err = np.array([r["std"] if r["count"] > 1 else 0 for r in rows])
        pos = x + (i - 2) * (w + gap)
        ax.bar(pos, y, w, color=color, edgecolor=INK, linewidth=1.5, hatch=hatch, label=label, zorder=2,
               yerr=err if key == "ft_minilm" else None, capsize=6,
               error_kw=dict(elinewidth=3, capthick=3, ecolor=INK))
        if key == "ft_minilm":
            for px, py, e in zip(pos, y, err):
                ax.text(px, py + e + 0.015, f"{py:.2f}", ha="center", va="bottom", fontsize=19,
                        fontweight="bold", color=INK)
    ax.set_xticks(x, [t[1] for t in tiers])
    ax.set_ylabel("Correct SOP ranked first (R@1)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=1.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"R@1 over 1,012 SOPs (fine-tuned: mean ± std, {n_ft} seeds)", loc="left", fontsize=20, color=INK2, pad=14)
    save(fig, "fig_retrieval")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(17, 5.0))
    ax.set_xlim(0, 17); ax.set_ylim(0.2, 5.15); ax.axis("off")

    def box(x, y, w, h, title, sub, learned):
        fc = "#dbe9fb" if learned else "#f2f1ed"
        ec = BLUE_STRONG if learned else INK2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fc, ec=ec, lw=3, zorder=2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=21, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=16, color=INK2, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=26, lw=3,
                                     color=INK, zorder=1))

    yb, h = 3.35, 1.7
    box(0.1, yb, 2.6, h, "Ticket", "incident\ntext", False)
    box(3.3, yb, 3.3, h, "Retriever", "fine-tuned MiniLM\nover 1,012 SOPs", True)
    box(7.2, yb, 2.9, h, "Planner", "one skill per\nSOP step (rules)", False)
    box(10.7, yb, 2.9, h, "Verifier", "pre/postconditions\nfrom sim state", False)
    box(14.2, yb, 2.7, h, "Verdict", "resolved /\npartial / failed", False)
    for x0, x1 in [(2.7, 3.3), (6.6, 7.2), (10.1, 10.7), (13.6, 14.2)]:
        arrow(x0, yb + h / 2, x1, yb + h / 2)

    # body skills under the executor
    skills = [("walk", "AMO policy\n(frozen)", True), ("pick/place", "IK + force-\ngated latch", False),
              ("press", "PPO +\ncurriculum", True), ("lever", "BC→DAgger→\nPPO (partial)", True)]
    sw, sh, sy = 2.55, 1.55, 0.35
    xs = [5.4 + i * (sw + 0.35) for i in range(4)]
    for (t, sub, learned), sx in zip(skills, xs):
        box(sx, sy, sw, sh, t, sub, learned)
    for i, sx in enumerate(xs):  # fan out from spread points on the verifier's bottom edge
        arrow(11.0 + i * 0.77, yb, xs[i] + sw / 2, sy + sh)
    ax.text(5.3, sy + sh / 2, "G1 skills\n(MuJoCo)", ha="right", va="center", fontsize=18, color=INK2)

    # legend
    ax.add_patch(FancyBboxPatch((0.15, 1.35), 0.55, 0.38, boxstyle="round,pad=0.02", fc="#dbe9fb", ec=BLUE_STRONG, lw=3))
    ax.text(0.85, 1.54, "learned", va="center", fontsize=17)
    ax.add_patch(FancyBboxPatch((0.15, 0.7), 0.55, 0.38, boxstyle="round,pad=0.02", fc="#f2f1ed", ec=INK2, lw=3))
    ax.text(0.85, 0.89, "rules / controller", va="center", fontsize=17)
    save(fig, "fig_pipeline")


def fig_pipeline_col():
    """Single-column version of the system diagram for the two-column paper (top right of page 1)."""
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")

    def box(x, y, w, h, title, sub, learned, tfs=22, sfs=16.5):
        fc = "#dbe9fb" if learned else "#f2f1ed"
        ec = BLUE_STRONG if learned else INK2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                                    fc=fc, ec=ec, lw=3, zorder=2))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=tfs, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color=INK2, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, lw=3,
                                     color=INK, zorder=1))

    w, h = 2.95, 1.75
    xs = [0.1, 3.52, 6.95]
    ya, yb = 5.7, 3.25
    box(xs[0], ya, w, h, "Ticket", "operator\ntext", False)
    box(xs[1], ya, w, h, "Retriever", "fine-tuned,\n1,012 SOPs", True)
    box(xs[2], ya, w, h, "Planner", "one skill per\nSOP step", False)
    arrow(xs[0] + w, ya + h / 2, xs[1], ya + h / 2)
    arrow(xs[1] + w, ya + h / 2, xs[2], ya + h / 2)
    box(xs[2], yb, w, h, "Verifier", "pre/post checks\nfrom sim state", False)
    box(xs[1], yb, w, h, "Verdict", "resolved /\npartial / failed", False)
    arrow(xs[2] + w / 2, ya, xs[2] + w / 2, yb + h)
    arrow(xs[2], yb + h / 2, xs[1] + w, yb + h / 2)

    skills = [("walk", "AMO\n(frozen)", True), ("pick", "IK + force\nlatch", False),
              ("press", "PPO +\ncurriculum", True), ("lever", "BC, DAgger,\nPPO (partial)", True)]
    sw, sh, sy = 2.3, 1.75, 0.15
    sxs = [0.1 + i * (sw + 0.2) for i in range(4)]
    for (t, sub, learned), sx in zip(skills, sxs):
        box(sx, sy, sw, sh, t, sub, learned, tfs=20, sfs=15.5)
    for i, sx in enumerate(sxs):
        arrow(xs[2] + 0.35 + i * 0.75, yb, sx + sw / 2, sy + sh)

    # legend in the free slot left of the verdict
    ax.add_patch(FancyBboxPatch((0.2, 4.5), 0.5, 0.36, boxstyle="round,pad=0.02", fc="#dbe9fb", ec=BLUE_STRONG, lw=3))
    ax.text(0.85, 4.68, "learned", va="center", fontsize=17)
    ax.add_patch(FancyBboxPatch((0.2, 3.75), 0.5, 0.36, boxstyle="round,pad=0.02", fc="#f2f1ed", ec=INK2, lw=3))
    ax.text(0.85, 3.93, "rules /\ncontroller", va="center", fontsize=17)
    save(fig, "fig_pipeline_col")


def fig_envelope(narrow=False):
    """narrow=True: single-column version for the two-column paper."""
    df = pd.read_csv(HERE / "data" / "action_envelope.csv")
    labels = [f"{t}:\n{j}" if narrow else f"{t}: {j}" for t, j in zip(df.task, df.joint)]
    fig, ax = plt.subplots(figsize=(8.6, 7.4) if narrow else (13.5, 5.6))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.required_offset_rad, height=0.55, color=BLUE_STRONG if narrow else BLUE_MID, edgecolor=INK, linewidth=1.5, zorder=2)
    for yy, v in zip(y, df.required_offset_rad):
        if narrow:  # inside the bar end, so labels never collide with the envelope lines
            ax.text(v - 0.03, yy, f"{v:.2f}", va="center", ha="right", fontsize=20, color="#ffffff", fontweight="bold")
        else:
            ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=19, color=INK)
    for x, style, lab in [(0.5, "--", "original envelope (0.5)"), (0.8, ":", "widened, press (0.8)"),
                          (1.2, "-.", "widened, lever (1.2)")]:
        ax.axvline(x, color=INK if x == 0.5 else INK2, linestyle=style, linewidth=3, label=lab, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Rest-to-contact joint offset (rad)")
    ax.set_xlim(0, 1.35)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=1.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02 if narrow else 0, 1.02), ncol=1 if narrow else 3, frameon=False,
              fontsize=20, handlelength=2.4, columnspacing=1.2)
    save(fig, "fig_envelope_col" if narrow else "fig_envelope")


if __name__ == "__main__":
    fig_retrieval()
    fig_pipeline()
    fig_pipeline_col()
    fig_envelope()
    fig_envelope(narrow=True)
