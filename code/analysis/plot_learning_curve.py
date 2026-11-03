"""Learning-curve figure: test score vs training days, old (100 ms, 1 s label) vs new (event-time, next mid change).

Reads results/learning_curve/{event,grid}.json; writes results/learning_curve.png.

    python code/analysis/plot_learning_curve.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SERIES = [("logreg", "Logistic regression", "#2a78d6"), ("lgbm", "LightGBM", "#eb6834"),
          ("mlp_small", "MLP small (4.5k)", "#1baf7a"), ("mlp_large", "MLP large (413k)", "#eda100")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    d = config.RESULTS_DIR / "learning_curve"
    ev, gr = (json.loads((d / f"{s}.json").read_text()) for s in ("event", "grid"))
    days = [1, 2, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, constrained_layout=True)
    # same 0.05 y-span on both panels so slopes are comparable
    panels = [(ev, "acc", "New: next mid change (event time), accuracy", ("rule_qi_sign", "Queue-imbalance rule"),
               (0.62, 0.67)),
              (gr, "macro_f1", "Old: mid over next 1 s (100 ms grid), macro-F1", None, (0.50, 0.55))]
    for ax, (r, metric, title, ref, ylim) in zip(axes, panels):
        ax.set_ylim(*ylim)
        ax.set_facecolor(SURFACE)
        for key, label, color in SERIES:
            ax.plot(days, [r[str(n)][key][metric] for n in days], color=color, lw=2, marker="o", ms=8,
                    mec=SURFACE, mew=2, label=label)
        if ref:
            ax.plot(days, [r[str(n)][ref[0]]["acc"] for n in days], color=MUTED, lw=1.5, ls="--", label=ref[1])
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Training days", color=MUTED)
        ax.set_xticks(days)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Changing the target helps far more than adding data (test day Dec 12 2025)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(config.RESULTS_DIR / "learning_curve.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
