"""Latency-aware market-making simulator (deliberately simple: it shows the latency effect, not real PnL).

Setting: a market maker quotes both sides of the book. At each decision row (a best bid/ask update on the
test day, Dec 12 2025) a model predicts the direction of the next mid change. If it predicts "up", the
maker pulls its ask (about to be run over); if "down", its bid. Payoffs, in ticks per decision:

    +1        prediction correct AND the pull lands before the adverse move (an adverse fill avoided)
    -c        prediction wrong AND the pull lands before the move (spread income / queue spot given up)
     0        the move happened before the pull landed (too late: same as having no model)

Timing. The pull lands at  t + model latency + own infrastructure latency (network, gateway, code), the
second swept on the x-axis. Each model's latency is its measured batch-1 CPU inference time
(results/latency/latency.json); hardware variants of DeepLOB use estimates and are labelled as such.

Competition. Optionally a rival reacts to the same book update after R microseconds with the same correct
signal and acts first if faster (e.g. picks off the stale quote). A correct pull is then only worth +1 if
it lands before min(next mid change, rival). Without a rival, the market's own clock (time to the next mid
change, measured in the data) is the deadline.

Assumptions (stated on every output): one lot; our orders do not move the market (non-reactive replay);
queue position and fees not modelled; decision rows are the thinned event-time rows (<= 300k per
stock-day), so results are reported per 1,000 decisions, not per day.

    python code/sim/latency_sim.py                      # all models with saved predictions
    python code/sim/latency_sim.py --cost 0.5 --rival-us 50

Inputs: $HFT_DATA_DIR/predictions/{test_rows.parquet, <model>_test.npy} (code/models/save_predictions.py)
and $HFT_DATA_DIR/deeplob_itch/p_up_test_seed0.npy (code/models/train_deeplob_itch.py).
Outputs: results/sim/latency_sim.json, results/sim/latency_sim.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PRED_DIR = config.DATA_DIR / "predictions"
OUT = config.RESULTS_DIR / "sim"
INFRA_MS = np.logspace(-3, 2, 61)  # own infrastructure latency swept: 1 us .. 100 ms

# Plotted models: (label, prediction file, latency key or fixed ms, colour, line style)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def model_latency_ms():
    lat = json.loads((config.RESULTS_DIR / "latency" / "latency.json").read_text())
    return {
        "mlp_small": lat["mlp_small"]["cpu1_torchscript"]["p50"] / 1e3,
        "mlp_large": lat["mlp_large"]["cpu1_torchscript"]["p50"] / 1e3,
        "lgbm": lat["lgbm"]["cpu1_python_predict"]["p50"] / 1e3,
        "deeplob": lat["deeplob"]["cpu1_eager"]["p50"] / 1e3,
        "logreg": 0.001,      # 34-term dot product: ~1 us, assumed
        "rule_qi": 0.0005,    # one ratio: assumed
    }


def load_models():
    rows = pl.read_parquet(PRED_DIR / "test_rows.parquet")
    preds = {p.stem.removesuffix("_test"): np.load(p) for p in PRED_DIR.glob("*_test.npy")}
    dl = config.DATA_DIR / "deeplob_itch" / "p_up_test_seed0.npy"
    if dl.exists():
        preds["deeplob"] = np.load(dl)
    for k, v in preds.items():
        assert len(v) == rows.height, (k, len(v), rows.height)
    return rows, preds


def value(p, y, dt_ms, latency_ms, cost, rival_ms=np.inf, threshold=0.0):
    """Mean payoff in ticks per 1,000 decisions, for each total latency in latency_ms (array)."""
    act = np.abs(p - 0.5) >= threshold
    correct = (p > 0.5) == (y == 1)
    dl_good = np.minimum(dt_ms, rival_ms)[act & correct]     # deadline for a useful pull
    dl_bad = dt_ms[act & ~correct]                           # a wrong pull costs only if it lands in time
    n = len(p)
    good = np.searchsorted(np.sort(dl_good), latency_ms, side="right")   # count with deadline <= latency
    bad = np.searchsorted(np.sort(dl_bad), latency_ms, side="right")
    n_good_in_time = len(dl_good) - good
    n_bad_in_time = len(dl_bad) - bad
    return 1000 * (n_good_in_time - cost * n_bad_in_time) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.5, help="ticks lost per wrong pull")
    ap.add_argument("--rival-us", type=float, default=50.0, help="rival reaction time for panel (b), microseconds")
    ap.add_argument("--threshold", type=float, default=0.0, help="act only if |P(up) - 0.5| >= threshold")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, preds = load_models()
    y = rows["y_next"].to_numpy()
    dt = rows["dt_next_ms"].to_numpy().astype(np.float64)
    lat = model_latency_ms()
    rival = args.rival_us / 1e3

    # (name, prediction key, inference latency ms, colour, style, label)
    series = [("oracle", None, 0.0, GREY, "--", "Perfect predictor, zero inference time")]
    if "mlp_small" in preds:
        series.append(("mlp_small", "mlp_small", lat["mlp_small"], BLUE, "--", "Small MLP 4.6k, direct (CPU, measured)"))
    if "mlp_small_kd_lgbm_a0.5" in preds:  # same 4.6k network, trained on LightGBM's soft labels (distill.py)
        series.append(("mlp_small_kd", "mlp_small_kd_lgbm_a0.5", lat["mlp_small"], BLUE, "-",
                       "Small MLP 4.6k, distilled (CPU, measured)"))
    if "lgbm" in preds:
        series.append(("lgbm", "lgbm", lat["lgbm"], ORANGE, "-", "LightGBM (CPU, measured)"))
    if "deeplob" in preds:
        series.append(("deeplob", "deeplob", lat["deeplob"], AQUA, "-", "DeepLOB (CPU, measured)"))
        series.append(("deeplob_h100", "deeplob", 0.1, AQUA, "--", "DeepLOB (H100 optimised, est. 0.1 ms)"))
        series.append(("deeplob_fpga", "deeplob", 0.002, AQUA, ":", "DeepLOB (FPGA streaming, est. 2 us)"))
    others = [k for k in ("mlp_large", "logreg", "rule_qi") if k in preds]

    res = dict(assumptions=dict(cost_ticks_per_wrong_pull=args.cost, rival_us=args.rival_us,
                                threshold=args.threshold, unit="ticks per 1,000 decisions",
                                notes="1 lot, non-reactive replay, no queue position or fees; test day Dec 12 2025, "
                                      "QQQ/PYPL/NVDA event-time rows"),
               infra_ms=INFRA_MS.tolist(), model_latency_ms=lat, models={})
    oracle_p = y.astype(np.float64)  # always right
    for name, key, inf_ms, *_ in series + [(k, k, lat[k]) for k in others]:
        p = oracle_p if key is None else preds[key]
        acc = float(((p > 0.5) == (y == 1)).mean())
        res["models"][name] = dict(
            inference_ms=inf_ms, accuracy=acc,
            no_rival=value(p, y, dt, inf_ms + INFRA_MS, args.cost, np.inf, args.threshold).tolist(),
            rival=value(p, y, dt, inf_ms + INFRA_MS, args.cost, rival, args.threshold).tolist())
    (OUT / "latency_sim.json").write_text(json.dumps(res, indent=1))

    # summary at a few infrastructure latencies
    probe = [0.01, 0.1, 1.0]
    idx = [int(np.argmin(np.abs(INFRA_MS - v))) for v in probe]
    print(f"ticks per 1,000 decisions (cost {args.cost}); columns = own infrastructure latency {probe} ms")
    for name, r in res["models"].items():
        a = " ".join(f"{r['no_rival'][i]:7.1f}" for i in idx)
        b = " ".join(f"{r['rival'][i]:7.1f}" for i in idx)
        print(f"  {name:13s} acc {r['accuracy']:.3f} inf {r['inference_ms'] * 1e3:8.1f} us | no rival {a} | "
              f"rival {args.rival_us:g} us {b}")
    plot(res, series, args)


def plot(res, series, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    x = np.array(res["infra_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200, sharey=True)
    tiers = [(1e-3, 1e-2, "FPGA / co-located\nhardware"), (1e-2, 0.3, "co-located\nsoftware"),
             (0.3, 5, "nearby data\ncentre"), (5, 100, "remote /\nretail")]
    titles = ["(a) Racing the market's clock (no rival)",
              f"(b) Rival reacts {args.rival_us:g} µs after the same update"]
    for ax, key, title in zip(axes, ["no_rival", "rival"], titles):
        for i, (lo, hi, lab) in enumerate(tiers):
            if i % 2 == 0:
                ax.axvspan(lo, hi, color="#f3f2ef", zorder=0, lw=0)
            ax.text(np.sqrt(lo * hi), 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color=MUTED)
        for name, _, inf_ms, col, ls, label in series:
            ax.plot(x, res["models"][name][key], color=col, ls=ls, lw=2, label=label, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-240, 1180)  # headroom so the tier labels clear the curves
        ax.set_xlabel("Own infrastructure latency (network, gateway, code), ms", color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_ylabel("Ticks saved per 1,000 decisions", color=MUTED)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.01, 0.005, f"Each model's measured (or estimated) inference time is added to the x value. Wrong pull "
             f"costs {args.cost:g} tick. 1 lot, non-reactive replay, no queue position or fees. Test day Dec 12 2025, "
             "QQQ / PYPL / NVDA.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "latency_sim.png")
    print("wrote", OUT / "latency_sim.png")


if __name__ == "__main__":
    main()
