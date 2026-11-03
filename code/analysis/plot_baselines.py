"""Figure comparing the baseline models: macro-F1 and directional accuracy vs prediction horizon.

Reads results/baselines/metrics.json and results/nn/metrics.json; writes results/baseline_comparison.png.

    python code/analysis/plot_baselines.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Validated categorical slots 1-4 (dataviz reference palette, light mode); reference line in muted gray.
SERIES = [("logreg", "Logistic regression", "#2a78d6"), ("lgbm", "LightGBM", "#eb6834"),
          ("small", "MLP small (4.5k)", "#1baf7a"), ("large", "MLP large (413k)", "#eda100")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    b = json.loads((config.RESULTS_DIR / "baselines/metrics.json").read_text())["results"]
    n = json.loads((config.RESULTS_DIR / "nn/metrics.json").read_text())["results"]
    hs = sorted(b, key=int)
    x = [int(h) / 10 for h in hs]

    def get(key, h):
        return b[h][key] if key in b[h] else n[h][key]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, constrained_layout=True)
    panels = [("macro_f1", "Macro-F1 (down / flat / up)", "prior"),
              ("dir_acc_when_both_move", "Directional accuracy when both move", None)]
    for ax, (metric, title, ref) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for key, label, color in SERIES:
            ax.plot(x, [get(key, h)[metric] for h in hs], color=color, lw=2, marker="o", ms=8,
                    mec=SURFACE, mew=2, label=label)
        if ref:
            ax.plot(x, [b[h][ref][metric] for h in hs], color=MUTED, lw=1.5, ls="--", label="Majority class")
        else:
            ax.axhline(0.5, color=MUTED, lw=1.5, ls="--", label="Coin flip")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Prediction horizon (s)", color=MUTED)
        ax.set_xticks(x)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("All models tie: test day Dec 12 2025, pooled QQQ / PYPL / NVDA", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.savefig(config.RESULTS_DIR / "baseline_comparison.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
