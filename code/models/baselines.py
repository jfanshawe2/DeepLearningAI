"""Linear and LightGBM baselines for mid-price direction prediction, pooled across stocks.

Chronological split (config.SPLIT): train Dec 8-10, validate Dec 11, test Dec 12 (2025). One model is
trained on all symbols together; the symbol id is an input (one-hot for the linear model, categorical for
LightGBM). Validation is used only for LightGBM early stopping.

Models, for each horizon h:
  prior     predict the training-set majority class
  logreg    multinomial logistic regression on standardised features
  lgbm      LightGBM multiclass
  ridge / lgbm_reg   regressions on dm_h, scored by information coefficient (corr with realised move)

Outputs in results/baselines/: metrics.json, metrics.md, lgbm_importance_h<h>.csv, importance figure.

    python code/models/baselines.py --horizons 10 50 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.features import feature_names, load  # noqa: E402

OUT = config.RESULTS_DIR / "baselines"


def design(df, cols, scaler=None, sym_levels=None):
    X = df.select(cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    onehot = (df["sym"].to_numpy()[:, None] == np.array(sym_levels)[None, :]).astype(np.float32)
    return np.hstack([Xs, onehot]), scaler


def class_metrics(y, pred, sym):
    out = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
               f1_per_class=f1_score(y, pred, average=None, labels=[0, 1, 2]).tolist(),
               confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    moved = (y != 1) & (pred != 1)
    out["dir_acc_when_both_move"] = float((y[moved] == pred[moved]).mean()) if moved.any() else None
    out["coverage_moves"] = float(moved.mean())
    out["per_symbol"] = {}
    for s in np.unique(sym):
        m = sym == s
        out["per_symbol"][config.CANDIDATES[s]] = dict(
            acc=accuracy_score(y[m], pred[m]), macro_f1=f1_score(y[m], pred[m], average="macro"))
    return out


def ic(y, pred, sym):
    out = dict(pearson=float(np.corrcoef(y, pred)[0, 1]), spearman=float(spearmanr(y, pred).statistic))
    out["per_symbol"] = {config.CANDIDATES[s]: float(np.corrcoef(y[sym == s], pred[sym == s])[0, 1])
                         for s in np.unique(sym)}
    return out


def run(h, tr, va, te, cols, stride):
    import lightgbm as lgb  # lazy: torch and lightgbm bundle clashing libomp copies on macOS
    ycol, dcol = f"y_{h}", f"dm_{h}"
    tr, va, te = (d.filter(pl.col(ycol) >= 0) for d in (tr, va, te))
    tr_s = tr.gather_every(stride)  # consecutive 100 ms rows are almost identical; thin for the linear fits
    syms = sorted(tr["sym"].unique().to_list())
    y_tr, y_te = tr[ycol].to_numpy(), te[ycol].to_numpy()
    sym_te = te["sym"].to_numpy()
    res = {"n_train": tr.height, "n_val": va.height, "n_test": te.height,
           "class_balance_train": np.bincount(y_tr, minlength=3).tolist(),
           "class_balance_test": np.bincount(y_te, minlength=3).tolist()}

    prior = np.bincount(y_tr, minlength=3).argmax()
    res["prior"] = class_metrics(y_te, np.full_like(y_te, prior), sym_te)

    t = time.time()
    lin_cols = [c for c in cols if c != "sym"]  # symbol enters the linear models as a one-hot instead
    X_tr, sc = design(tr_s, lin_cols, sym_levels=syms)
    X_te, _ = design(te, lin_cols, sc, syms)
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_tr, tr_s[ycol].to_numpy())
    res["logreg"] = class_metrics(y_te, lr.predict(X_te), sym_te)
    rg = Ridge(alpha=1.0).fit(X_tr, tr_s[dcol].to_numpy())
    res["ridge"] = ic(te[dcol].to_numpy(), rg.predict(X_te), sym_te)
    res["linear_seconds"] = time.time() - t

    t = time.time()
    feats = cols  # includes sym, used as a LightGBM categorical
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=8)
    dtr = lgb.Dataset(tr.select(feats).to_pandas(), y_tr, categorical_feature=["sym"], free_raw_data=False)
    dva = lgb.Dataset(va.select(feats).to_pandas(), va[ycol].to_numpy(), reference=dtr)
    clf = lgb.train(params, dtr, 2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    proba = clf.predict(te.select(feats).to_pandas(), num_iteration=clf.best_iteration)
    res["lgbm"] = class_metrics(y_te, proba.argmax(1), sym_te)
    res["lgbm"]["best_iter"] = clf.best_iteration
    res["lgbm"]["test_logloss"] = float(-np.mean(np.log(proba[np.arange(len(y_te)), y_te] + 1e-12)))

    rparams = dict(objective="huber", alpha=1.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                   feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=8)
    rtr = lgb.Dataset(tr.select(feats).to_pandas(), tr[dcol].to_numpy(), categorical_feature=["sym"])
    rva = lgb.Dataset(va.select(feats).to_pandas(), va[dcol].to_numpy(), reference=rtr)
    reg = lgb.train(rparams, rtr, 2000, valid_sets=[rva],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    res["lgbm_reg"] = ic(te[dcol].to_numpy(), reg.predict(te.select(feats).to_pandas(),
                                                         num_iteration=reg.best_iteration), sym_te)
    res["lgbm_seconds"] = time.time() - t

    imp = pl.DataFrame({"feature": feats, "gain": clf.feature_importance("gain")}).sort("gain", descending=True)
    imp.write_csv(OUT / f"lgbm_importance_h{h}.csv")
    return res


def write_markdown(all_res, symbols):
    lines = [f"# Baselines: mid-price direction, symbols {', '.join(symbols)}", "",
             "Train Dec 8-10 2025, validate Dec 11 (LightGBM early stopping only), test Dec 12. "
             "Classes: down / flat / up in mid over the horizon.", "",
             "| Horizon | Model | Accuracy | Macro-F1 | Dir. acc. (both move) | IC (regression) |",
             "|---|---|---|---|---|---|"]
    for h, r in all_res.items():
        sec = int(h) / 10
        for m, regm in [("prior", None), ("logreg", "ridge"), ("lgbm", "lgbm_reg")]:
            d = r[m]
            da = f"{d['dir_acc_when_both_move']:.3f}" if d["dir_acc_when_both_move"] is not None else "-"
            icv = f"{r[regm]['pearson']:.3f}" if regm else "-"
            lines.append(f"| {sec:g}s | {m} | {d['acc']:.3f} | {d['macro_f1']:.3f} | {da} | {icv} |")
    lines += ["", "Per-symbol macro-F1 (LightGBM):", "", "| Horizon | " + " | ".join(symbols) + " |",
              "|---|" + "---|" * len(symbols)]
    for h, r in all_res.items():
        ps = r["lgbm"]["per_symbol"]
        lines.append(f"| {int(h) / 10:g}s | " + " | ".join(f"{ps[s]['macro_f1']:.3f}" for s in symbols) + " |")
    lines += ["", "Test class balance (down / flat / up):", ""]
    for h, r in all_res.items():
        b = np.array(r["class_balance_test"]) / sum(r["class_balance_test"])
        lines.append(f"- {int(h) / 10:g}s: {b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f}")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")


def plot_importance(horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, horizons):
        imp = pl.read_csv(OUT / f"lgbm_importance_h{h}.csv").head(15)
        g = imp["gain"].to_numpy()
        ax.barh(imp["feature"].to_list()[::-1], (g / g.sum())[::-1], color="#4C72B0")
        ax.set_title(f"LightGBM gain, {h / 10:g}s horizon")
        ax.set_xlabel("share of top-15 gain")
    fig.savefig(OUT / "lgbm_importance.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    p.add_argument("--stride", type=int, default=5, help="row thinning for the linear models")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tr = load(config.SPLIT["train"], args.symbols)
    va = load(config.SPLIT["val"], args.symbols)
    te = load(config.SPLIT["test"], args.symbols)
    cols = feature_names(tr.drop("day"))
    print(f"train {tr.height:,} / val {va.height:,} / test {te.height:,} rows, {len(cols)} features")

    all_res = {}
    for h in args.horizons:
        t = time.time()
        all_res[str(h)] = run(h, tr, va, te, cols, args.stride)
        r = all_res[str(h)]
        print(f"h={h}: prior acc {r['prior']['acc']:.3f} | logreg acc {r['logreg']['acc']:.3f} "
              f"F1 {r['logreg']['macro_f1']:.3f} | lgbm acc {r['lgbm']['acc']:.3f} F1 {r['lgbm']['macro_f1']:.3f} "
              f"| IC ridge {r['ridge']['pearson']:.3f} lgbm {r['lgbm_reg']['pearson']:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
    (OUT / "metrics.json").write_text(json.dumps({"symbols": args.symbols, "features": cols,
                                                  "results": all_res}, indent=1))
    write_markdown(all_res, args.symbols)
    plot_importance(args.horizons)


if __name__ == "__main__":
    main()
