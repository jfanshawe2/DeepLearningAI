"""Save per-row test predictions P(up) of the event-time models, for the latency simulator.

Same rows, label and split as results/learning_curve/event.json (train Dec 8-10, validate Dec 11, test
Dec 12; label y_next). Row order = features_event.load(test day, config.SYMBOLS), which is also the order
train_deeplob_itch.py uses, so all prediction files line up row by row.

    python code/models/save_predictions.py --family trees      # logistic, LightGBM, queue-imbalance rule
    python code/models/save_predictions.py --family mlp        # MLP small / large (CPU; separate process:
                                                               # torch and lightgbm libomp clash on macOS)

Writes $HFT_DATA_DIR/predictions/<model>_test.npy (float32 P(up)) and test_rows.parquet (ts, sym, y_next,
dt_next_ms, spread) once.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import features_event as fe  # noqa: E402

PRED_DIR = config.DATA_DIR / "predictions"


def matrix(df, cols, mean=None, std=None):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-6
    return (X - mean) / std, mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["trees", "mlp"], required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tr = fe.load(config.SPLIT["train"], config.SYMBOLS)
    va = fe.load(config.SPLIT["val"], config.SYMBOLS)
    te = fe.load(config.SPLIT["test"], config.SYMBOLS)
    cols = [c for c in fe.feature_names(tr) if c != "sym"]
    y_tr, y_va, y_te = (d["y_next"].to_numpy().astype(np.int64) for d in (tr, va, te))
    rows = PRED_DIR / "test_rows.parquet"
    if not rows.exists():
        te.select("ts", "sym", "y_next", "dt_next_ms", "spread").write_parquet(rows)
    out = {}

    if args.family == "trees":
        import lightgbm as lgb
        q = te["qi1"].to_numpy()
        out["rule_qi"] = np.clip(0.5 + 0.5 * q, 0, 1)            # sign rule; |qi| as a crude confidence
        syms = sorted(tr["sym"].unique().to_list())
        oh = lambda d: (d["sym"].to_numpy()[:, None] == np.array(syms)[None, :]).astype(np.float32)  # noqa: E731
        Xtr, m, s = matrix(tr, cols)
        Xte, _, _ = matrix(te, cols, m, s)
        t = time.time()
        lr = LogisticRegression(max_iter=500).fit(np.hstack([Xtr, oh(tr)]), y_tr)
        out["logreg"] = lr.predict_proba(np.hstack([Xte, oh(te)]))[:, 1]
        print(f"logreg {time.time() - t:.0f}s", flush=True)
        t = time.time()
        params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                      feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      num_threads=args.threads)
        f = cols + ["sym"]
        dtr = lgb.Dataset(tr.select(f).to_pandas(), y_tr, categorical_feature=["sym"])
        dva = lgb.Dataset(va.select(f).to_pandas(), y_va, reference=dtr)
        mdl = lgb.train(params, dtr, 3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        out["lgbm"] = mdl.predict(te.select(f).to_pandas(), num_iteration=mdl.best_iteration)
        print(f"lgbm {mdl.best_iteration} trees, {time.time() - t:.0f}s", flush=True)
    else:
        import torch
        from models import mlp
        torch.set_num_threads(args.threads)
        Xtr, m, s = matrix(tr, cols)
        Xva, _, _ = matrix(va, cols, m, s)
        Xte, _, _ = matrix(te, cols, m, s)
        sidx = {v: i for i, v in enumerate(sorted(tr["sym"].unique().to_list()))}
        T = lambda X, d, y: (torch.from_numpy(X), torch.tensor([sidx[v] for v in d["sym"].to_list()]),  # noqa: E731
                             torch.from_numpy(y))
        Atr, Ava, Ate = T(Xtr, tr, y_tr), T(Xva, va, y_va), T(Xte, te, y_te)
        for size in ("small", "large"):
            t = time.time()
            model, hist = mlp.train(size, Atr, Ava, len(cols), len(sidx), "cpu", 20, 1e-3, 4096, n_out=2,
                                    verbose=False)
            out[f"mlp_{size}"] = mlp.predict(model, Ate[0], Ate[1], "cpu")[:, 1]
            print(f"mlp_{size} {len(hist)} epochs, {time.time() - t:.0f}s", flush=True)

    for k, v in out.items():
        np.save(PRED_DIR / f"{k}_test.npy", v.astype(np.float32))
        print(f"  {k}: test acc {accuracy_score(y_te, (v > 0.5).astype(int)):.4f}", flush=True)


if __name__ == "__main__":
    main()
