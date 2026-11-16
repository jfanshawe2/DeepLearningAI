"""Figure: our DeepLOB reproduction vs the paper's Table II (weighted F1, FI-2010 Setup 2).

Uses every results/deeplob_k<k>[_seed<n>].json for the decimal-precision runs; with several seeds the bar
is the mean and the whisker the min-max range.

    python deeplob_reproduction/plot_results.py      ->  results/fi2010_vs_paper.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PAPER = {10: 83.40, 20: 72.82, 50: 80.35}
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ours(k):
    runs = sorted(RES.glob(f"deeplob_k{k}.json")) + sorted(RES.glob(f"deeplob_k{k}_seed*.json"))
    return np.array([json.loads(p.read_text())["test"]["weighted"]["f1"] for p in runs])


def main():
    ks = [10, 20, 50]
    x = np.arange(len(ks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    ax.bar(x - w / 2 - 0.01, [PAPER[k] for k in ks], w, color=ORANGE, label="Paper (Table II)", zorder=2)
    vals = [ours(k) for k in ks]
    means = [v.mean() for v in vals]
    ax.bar(x + w / 2 + 0.01, means, w, color=BLUE, label="Our reproduction", zorder=2)
    if any(len(v) > 1 for v in vals):
        lo = [m - v.min() for m, v in zip(means, vals)]
        hi = [v.max() - m for m, v in zip(means, vals)]
        ax.errorbar(x + w / 2 + 0.01, means, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, lw=1, zorder=3)
    for i, k in enumerate(ks):
        ax.text(x[i] - w / 2 - 0.01, PAPER[k] + 0.6, f"{PAPER[k]:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(x[i] + w / 2 + 0.01, means[i] + 0.6, f"{means[i]:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    n_seeds = max(len(v) for v in vals)
    ax.set_xticks(x, [f"k = {k}" for k in ks])
    ax.set_ylim(50, 90)
    ax.set_ylabel("Weighted F1 (%)", color=MUTED)
    ax.set_title("DeepLOB on FI-2010 (Setup 2): reproduction vs paper"
                 + (f" ({n_seeds} seeds, min-max)" if n_seeds > 1 else ""), color=INK, fontsize=11, loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.text(0.01, 0.01, "Prediction horizon k in events. Test days 8-10, never used for training or early stopping. "
             "Axis starts at 50.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "fi2010_vs_paper.png")
    print("wrote", RES / "fi2010_vs_paper.png")


if __name__ == "__main__":
    main()
