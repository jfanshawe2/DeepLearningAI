"""Distil slow teachers into the 4.6k-parameter MLP (soft-label KD, Hinton et al. 2015; plan item D1).

Same rows, label and split as the event-time results: train Dec 8-10, validate Dec 11, test Dec 12.
The student is the small MLP of mlp.py (33 features + 4-d symbol embedding -> 64 -> 32 -> 2, 4,610 params).

Target per training row:   alpha * onehot(y_next) + (1 - alpha) * [1 - p_teacher, p_teacher]
with soft cross-entropy. alpha = 1 is the direct (no-teacher) baseline. Early stopping always uses the
validation loss against the TRUE labels, so every variant is selected the same way. Test day used once.

Teachers (P(up) on every train / val row):
  deeplob   DeepLOB trained on the raw book (train_deeplob_itch.py). Its train-row outputs are in-sample;
            it barely overfits (train loss 0.59, val 0.60), but this is noted.
  lgbm      LightGBM on the same features, CROSS-FITTED on the training days (for each day, a model trained
            on the other two days predicts it), so its soft labels are not over-confident in-sample fits.
            Validation rows: model trained on all three days.
  ens       average of deeplob and lgbm.

    python code/models/distill.py --stage teacher_lgbm      # LightGBM cross-fitted outputs (separate process:
                                                            # torch and lightgbm libomp clash on macOS)
    python code/models/distill.py --stage distill           # all variants x seeds, CPU

Outputs: results/distill/metrics.json; test P(up) of each variant (seed 0) in $HFT_DATA_DIR/predictions/
as mlp_small_kd_<teacher>_a<alpha>_test.npy (the latency simulator picks up the best one).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import features_event as fe  # noqa: E402

PRED_DIR = config.DATA_DIR / "predictions"
TEACH_DIR = config.DATA_DIR / "teachers"
OUT = config.RESULTS_DIR / "distill"
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def matrix(df, cols, mean=None, std=None):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-6
    return (X - mean) / std, mean, std


def teacher_lgbm(threads):
    import lightgbm as lgb
    TEACH_DIR.mkdir(parents=True, exist_ok=True)
    days = config.SPLIT["train"]
    parts = [fe.load([d], config.SYMBOLS) for d in days]
    va = fe.load(config.SPLIT["val"], config.SYMBOLS)
    f = [c for c in fe.feature_names(parts[0]) if c != "sym"] + ["sym"]
    params = dict(LGB_PARAMS, num_threads=threads)
    n_trees = 386  # chosen by early stopping on validation for the 3-day model (event.json)
    oof = []
    for i, d in enumerate(days):
        tr = pl.concat([p for j, p in enumerate(parts) if j != i])
        m = lgb.train(params, lgb.Dataset(tr.select(f).to_pandas(), tr["y_next"].to_numpy(),
                                          categorical_feature=["sym"]), n_trees)
        oof.append(m.predict(parts[i].select(f).to_pandas()))
        print(f"  cross-fit: predicted {d} from the other two days", flush=True)
    tr = pl.concat(parts)
    m = lgb.train(params, lgb.Dataset(tr.select(f).to_pandas(), tr["y_next"].to_numpy(),
                                      categorical_feature=["sym"]), n_trees)
    np.save(TEACH_DIR / "lgbm_train_oof.npy", np.concatenate(oof).astype(np.float32))
    np.save(TEACH_DIR / "lgbm_val.npy", m.predict(va.select(f).to_pandas()).astype(np.float32))
    print("saved LightGBM teacher outputs", flush=True)


def soft_ce(logits, target):
    import torch
    return -(target * torch.log_softmax(logits, 1)).sum(1).mean()


def train_student(A_tr, soft_tr, A_va, n_in, n_sym, alpha, seed, epochs=20, lr=1e-3, bs=4096):
    """Small MLP with soft targets; early stopping on validation loss vs the true labels (patience 3)."""
    import torch
    import torch.nn.functional as F
    from models.mlp import MLP, SIZES
    torch.manual_seed(seed)
    X, s, y = A_tr
    tgt = alpha * F.one_hot(y, 2).float() + (1 - alpha) * torch.stack([1 - soft_tr, soft_tr], 1)
    model = MLP(n_in, SIZES["small"], n_sym, n_out=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            soft_ce(model(X[idx], s[idx]), tgt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(A_va[0], A_va[1]), A_va[2]).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


def distill(threads, seeds, alphas):
    import torch
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    torch.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = (fe.load(config.SPLIT[k], config.SYMBOLS) for k in ("train", "val", "test"))
    cols = [c for c in fe.feature_names(tr) if c != "sym"]
    Xtr, m, s = matrix(tr, cols)
    Xva, _, _ = matrix(va, cols, m, s)
    Xte, _, _ = matrix(te, cols, m, s)
    sidx = {v: i for i, v in enumerate(sorted(tr["sym"].unique().to_list()))}
    A = lambda X, d: (torch.from_numpy(X), torch.tensor([sidx[v] for v in d["sym"].to_list()]),  # noqa: E731
                      torch.from_numpy(d["y_next"].to_numpy().astype(np.int64)))
    A_tr, A_va, A_te = A(Xtr, tr), A(Xva, va), A(Xte, te)
    y_te = te["y_next"].to_numpy()

    dl = config.DATA_DIR / "deeplob_itch"
    teachers = {"deeplob": (np.load(dl / "p_up_train_seed0.npy"), np.load(dl / "p_up_val_seed0.npy")),
                "lgbm": (np.load(TEACH_DIR / "lgbm_train_oof.npy"), np.load(TEACH_DIR / "lgbm_val.npy"))}
    teachers["ens"] = tuple((a + b) / 2 for a, b in zip(teachers["deeplob"], teachers["lgbm"]))
    for k, (pt, pv) in teachers.items():
        assert len(pt) == len(tr) and len(pv) == len(va), k
    teacher_info = {k: dict(train_acc=float(accuracy_score(tr["y_next"].to_numpy(), pt > 0.5)),
                            val_acc=float(accuracy_score(va["y_next"].to_numpy(), pv > 0.5)))
                    for k, (pt, pv) in teachers.items()}
    print("teachers:", teacher_info, flush=True)

    variants = [("direct", None, 1.0)] + [(f"kd_{t}_a{a:g}", t, a) for t in teachers for a in alphas]
    res = dict(teachers=teacher_info, seeds=seeds, alphas=alphas, variants={})
    for name, t, alpha in variants:
        soft = torch.zeros(len(tr)) if t is None else torch.from_numpy(teachers[t][0].astype(np.float32))
        runs = []
        for seed in seeds:
            t0 = time.time()
            model, n_ep, vbest = train_student(A_tr, soft, A_va, len(cols), len(sidx), alpha, seed)
            with torch.no_grad():
                p = torch.softmax(model(A_te[0], A_te[1]), 1)[:, 1].numpy()
            runs.append(dict(seed=seed, acc=float(accuracy_score(y_te, p > 0.5)), auc=float(roc_auc_score(y_te, p)),
                             logloss=float(log_loss(y_te, np.c_[1 - p, p].clip(1e-7, 1))), epochs=n_ep,
                             val_loss=vbest, seconds=time.time() - t0))
            if seed == seeds[0]:
                key = "mlp_small_direct" if t is None else f"mlp_small_{name}"
                np.save(PRED_DIR / f"{key}_test.npy", p.astype(np.float32))
        accs = np.array([r["acc"] for r in runs])
        res["variants"][name] = dict(teacher=t, alpha=alpha, runs=runs, acc_mean=float(accs.mean()),
                                     acc_min=float(accs.min()), acc_max=float(accs.max()),
                                     auc_mean=float(np.mean([r["auc"] for r in runs])),
                                     logloss_mean=float(np.mean([r["logloss"] for r in runs])))
        v = res["variants"][name]
        print(f"  {name:16s} acc {v['acc_mean']:.4f} [{v['acc_min']:.4f}, {v['acc_max']:.4f}] auc {v['auc_mean']:.4f} "
              f"logloss {v['logloss_mean']:.4f}", flush=True)
        (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["teacher_lgbm", "distill"], required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.0])
    args = p.parse_args()
    if args.stage == "teacher_lgbm":
        teacher_lgbm(args.threads)
    else:
        distill(args.threads, args.seeds, args.alphas)


if __name__ == "__main__":
    main()
