"""Does more training data help? Train on 1, 2 and 3 days and score on the same test day.

Two setups:
  grid   100 ms snapshots, label y_10 (mid over the next 1 s: down / flat / up)          -- original
  event  best bid/ask updates, label y_next (direction of the next mid change: down / up) -- new

Training sets end on the day before validation, so the smallest set is the most recent day:
  1 day = Dec 10, 2 days = Dec 9-10, 3 days = Dec 8-10. Validation Dec 11 (early stopping), test Dec 12.

Models: logistic regression, LightGBM, MLP small (4.5k params), MLP large (413k). For the event setup
a one-feature rule (sign of queue imbalance) is included as a reference.

    python code/models/learning_curve.py --setup event --models logreg lgbm
    python code/models/learning_curve.py --setup event --models mlp_small mlp_large

LightGBM and PyTorch each bundle their own libomp on macOS and segfault/hang when loaded in one process,
so torch is imported only when an MLP is requested; run tree and neural models as separate invocations.
Results are merged into results/learning_curve/<setup>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import features as grid_feats  # noqa: E402
from data import features_event as event_feats  # noqa: E402

OUT = config.RESULTS_DIR / "learning_curve"
TRAIN_SETS = {1: ["2025-12-10"], 2: ["2025-12-09", "2025-12-10"], 3: ["2025-12-08", "2025-12-09", "2025-12-10"]}


def setup(name):
    if name == "grid":
        return dict(load=grid_feats.load, names=lambda df: grid_feats.feature_names(df.drop("day")),
                    label="y_10", n_cls=3, lin_stride=5)
    return dict(load=event_feats.load, names=event_feats.feature_names, label="y_next", n_cls=2, lin_stride=1)


def matrix(df, cols, mean=None, std=None):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-6
    return (X - mean) / std, mean, std


def score(y, proba, sym, n_cls):
    pred = proba.argmax(1)
    r = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
             logloss=log_loss(y, np.clip(proba, 1e-7, 1), labels=list(range(n_cls))))
    if n_cls == 2:
        r["auc"] = roc_auc_score(y, proba[:, 1])
    r["per_symbol_acc"] = {config.CANDIDATES[s]: accuracy_score(y[sym == s], pred[sym == s]) for s in np.unique(sym)}
    return r


def run_models(tr, va, te, S, models, device):
    ycol, n_cls = S["label"], S["n_cls"]
    tr, va, te = (d.filter(pl.col(ycol) >= 0) for d in (tr, va, te))
    cols = [c for c in S["names"](tr) if c != "sym"]
    syms = sorted(tr["sym"].unique().to_list())
    sym_te = te["sym"].to_numpy()
    y_tr, y_va, y_te = (d[ycol].to_numpy().astype(np.int64) for d in (tr, va, te))
    Xtr, mean, std = matrix(tr, cols)
    Xva, _, _ = matrix(va, cols, mean, std)
    Xte, _, _ = matrix(te, cols, mean, std)
    oh = lambda d: (d["sym"].to_numpy()[:, None] == np.array(syms)[None, :]).astype(np.float32)  # noqa: E731
    out = {}

    if "logreg" in models:
        t = time.time()
        k = S["lin_stride"]
        lr = LogisticRegression(max_iter=500).fit(np.hstack([Xtr, oh(tr)])[::k], y_tr[::k])
        out["logreg"] = score(y_te, lr.predict_proba(np.hstack([Xte, oh(te)])), sym_te, n_cls)
        out["logreg"]["seconds"] = time.time() - t

    if "lgbm" in models:
        import lightgbm as lgb
        t = time.time()
        params = dict(objective="multiclass" if n_cls > 2 else "binary", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
                      lambda_l2=1.0, verbose=-1, num_threads=8)
        if n_cls > 2:
            params["num_class"] = n_cls
        f = cols + ["sym"]
        dtr = lgb.Dataset(tr.select(f).to_pandas(), y_tr, categorical_feature=["sym"])
        dva = lgb.Dataset(va.select(f).to_pandas(), y_va, reference=dtr)
        m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(te.select(f).to_pandas(), num_iteration=m.best_iteration)
        p = np.c_[1 - p, p] if n_cls == 2 else p
        out["lgbm"] = score(y_te, p, sym_te, n_cls)
        out["lgbm"].update(best_iter=m.best_iteration, seconds=time.time() - t)
        imp = sorted(zip(f, m.feature_importance("gain")), key=lambda z: -z[1])[:10]
        out["lgbm"]["top_features"] = [n for n, _ in imp]

    if not any(m.startswith("mlp") for m in models):
        return add_rule(out, te, y_te, n_cls)
    import torch
    from models import mlp
    sidx = {s: i for i, s in enumerate(syms)}
    T = lambda X, d, y: (torch.from_numpy(X), torch.tensor([sidx[v] for v in d["sym"].to_list()]),  # noqa: E731
                         torch.from_numpy(y))
    Atr, Ava, Ate = T(Xtr, tr, y_tr), T(Xva, va, y_va), T(Xte, te, y_te)
    for size in ("small", "large"):
        if f"mlp_{size}" not in models:
            continue
        t = time.time()
        model, hist = mlp.train(size, Atr, Ava, len(cols), len(syms), device, 20, 1e-3, 4096,
                                n_out=n_cls, verbose=False)
        p = mlp.predict(model, Ate[0], Ate[1], device)
        out[f"mlp_{size}"] = score(y_te, p, sym_te, n_cls)
        out[f"mlp_{size}"].update(epochs=len(hist), seconds=time.time() - t)

    return add_rule(out, te, y_te, n_cls)


def add_rule(out, te, y_te, n_cls):
    if n_cls == 2:
        q = te["qi1"].to_numpy()
        out["rule_qi_sign"] = dict(acc=float(((q > 0).astype(int) == y_te).mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", choices=["grid", "event"], required=True)
    p.add_argument("--models", nargs="+", default=["logreg", "lgbm", "mlp_small", "mlp_large"])
    p.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = setup(args.setup)
    device = "cpu"
    if any(m.startswith("mlp") for m in args.models):
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = OUT / f"{args.setup}.json"
    va = S["load"](config.SPLIT["val"], config.SYMBOLS)
    te = S["load"](config.SPLIT["test"], config.SYMBOLS)
    res = json.loads(path.read_text()) if path.exists() else {}
    for n in args.sizes:
        tr = S["load"](TRAIN_SETS[n], config.SYMBOLS)
        t = time.time()
        new = run_models(tr, va, te, S, args.models, device)
        res.setdefault(str(n), {}).update(new)
        res[str(n)]["n_train"] = tr.height
        line = " | ".join(f"{k} acc {v['acc']:.4f}" + (f" auc {v['auc']:.4f}" if "auc" in v else "")
                          + (f" F1 {v['macro_f1']:.4f}" if "macro_f1" in v else "")
                          for k, v in new.items() if isinstance(v, dict))
        print(f"[{args.setup}] {n} day(s), {tr.height:,} rows ({time.time() - t:.0f}s): {line}", flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
