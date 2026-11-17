"""Students with a raw-book window (students.py), trained directly and distilled; follow-up to distill.py.

Question: if the student can see (a compact slice of) the raw book, can it absorb DeepLOB's raw-book edge
through distillation, and is the extra accuracy worth the extra microseconds?

Same rows, label, split, teachers and selection rule as distill.py (early stopping on validation loss vs the
true labels; test day once; 3 seeds). Teachers: DeepLOB, LightGBM (cross-fitted), their average; alpha 0.5.

Raw window per row: the last K = 10 updates of the top L = 5 levels from the full update stream
(book_events), prices in ticks relative to the row's current mid (clipped to +-20), sizes z-scored with the
previous day's statistics (as for DeepLOB). Built once and cached as float16 in $HFT_DATA_DIR/raw_student/.

    python code/models/distill_raw.py --stage build      # raw windows (~11 GB RAM peak while loading books)
    python code/models/distill_raw.py --stage train      # 2 students x 4 variants x 3 seeds, CPU

Outputs: results/distill_raw/metrics.json; test P(up) (seed 0) in $HFT_DATA_DIR/predictions/
as student_<arch>_<variant>_test.npy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data import features_event as fe  # noqa: E402

RAW_DIR = config.DATA_DIR / "raw_student"
PRED_DIR = config.DATA_DIR / "predictions"
TEACH_DIR = config.DATA_DIR / "teachers"
OUT = config.RESULTS_DIR / "distill_raw"


def build():
    from models.students import K, L
    from models.train_deeplob_itch import PRICE, Split
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cols = np.arange(4 * L)                       # levels 1..L in COLS order (ap, as, bp, bs per level)
    pcols = cols[PRICE[cols]]                     # ap_i, bp_i -> positions in the int16 price array
    pidx = np.cumsum(PRICE) - 1                   # column -> index among price columns
    sidx = np.cumsum(~PRICE) - 1
    p_sel, s_sel = pidx[pcols], sidx[cols[~PRICE[cols]]]
    prev = {}
    for name in ("train", "val", "test"):
        sp = Split(config.SPLIT[name], prev)
        n = len(sp)
        out = np.empty((n, K, 4 * L), np.float16)
        offs = np.arange(-K + 1, 1)
        is_p = PRICE[cols]
        for i in range(0, n, 200_000):
            g = sp.gidx[i:i + 200_000][:, None] + offs[None, :]           # (b, K)
            P = sp.P[g][:, :, p_sel].astype(np.float32)                      # (b, K, 2L) cents rel. to ref
            mid = (P[:, -1, 0] + P[:, -1, 1]) / 2                            # ap1, bp1 of the current update
            w = np.empty((len(g), K, 4 * L), np.float32)
            w[:, :, is_p] = np.clip(P - mid[:, None, None], -20, 20)
            w[:, :, ~is_p] = sp.S[g][:, :, s_sel]
            out[i:i + 200_000] = w
        y_ref = fe.load(config.SPLIT[name], config.SYMBOLS)["y_next"].to_numpy()
        assert (y_ref == sp.y).all(), f"row order mismatch in {name}"
        np.save(RAW_DIR / f"{name}.npy", out)
        print(f"  {name}: {out.shape} saved", flush=True)
        del sp, out


def train(threads, seeds, alpha):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from models.distill import matrix, soft_ce
    from models.students import STUDENTS
    torch.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = (fe.load(config.SPLIT[k], config.SYMBOLS) for k in ("train", "val", "test"))
    cols = [c for c in fe.feature_names(tr) if c != "sym"]
    Xtr, m, s = matrix(tr, cols)
    Xva, _, _ = matrix(va, cols, m, s)
    Xte, _, _ = matrix(te, cols, m, s)
    R = {k: torch.from_numpy(np.load(RAW_DIR / f"{k}.npy").astype(np.float32)) for k in ("train", "val", "test")}
    sidx = {v: i for i, v in enumerate(sorted(tr["sym"].unique().to_list()))}
    S = lambda d: torch.tensor([sidx[v] for v in d["sym"].to_list()])  # noqa: E731
    Y = lambda d: torch.from_numpy(d["y_next"].to_numpy().astype(np.int64))  # noqa: E731
    D = {"train": (torch.from_numpy(Xtr), R["train"], S(tr), Y(tr)),
         "val": (torch.from_numpy(Xva), R["val"], S(va), Y(va)),
         "test": (torch.from_numpy(Xte), R["test"], S(te), Y(te))}
    y_te = te["y_next"].to_numpy()
    dl = config.DATA_DIR / "deeplob_itch"
    p_dl, p_lg = np.load(dl / "p_up_train_seed0.npy"), np.load(TEACH_DIR / "lgbm_train_oof.npy")
    teachers = {"deeplob": p_dl, "lgbm": p_lg, "ens": (p_dl + p_lg) / 2}

    res = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    for arch, cls in STUDENTS.items():
        for var, t in [("direct", None)] + [(f"kd_{k}", k) for k in teachers]:
            a = 1.0 if t is None else alpha
            X, raw, sy, y = D["train"]
            soft = torch.zeros(len(y)) if t is None else torch.from_numpy(teachers[t].astype(np.float32))
            tgt = a * F.one_hot(y, 2).float() + (1 - a) * torch.stack([1 - soft, soft], 1)
            runs = []
            for seed in seeds:
                t0 = time.time()
                torch.manual_seed(seed)
                model = cls(len(cols), len(sidx))
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
                best, best_state, bad, ep = np.inf, None, 0, 0
                for ep in range(20):
                    model.train()
                    perm = torch.randperm(len(y))
                    for i in range(0, len(perm), 4096):
                        idx = perm[i:i + 4096]
                        opt.zero_grad()
                        soft_ce(model(X[idx], raw[idx], sy[idx]), tgt[idx]).backward()
                        opt.step()
                    sched.step()
                    model.eval()
                    with torch.no_grad():
                        vl = F.cross_entropy(model(*D["val"][:3]), D["val"][3]).item()
                    if vl < best - 1e-5:
                        best, bad = vl, 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= 3:
                            break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    p = torch.softmax(model(*D["test"][:3]), 1)[:, 1].numpy()
                runs.append(dict(seed=seed, acc=float(accuracy_score(y_te, p > 0.5)),
                                 auc=float(roc_auc_score(y_te, p)),
                                 logloss=float(log_loss(y_te, np.c_[1 - p, p].clip(1e-7, 1))),
                                 epochs=ep + 1, val_loss=best, seconds=time.time() - t0))
                if seed == seeds[0]:
                    np.save(PRED_DIR / f"student_{arch}_{var}_test.npy", p.astype(np.float32))
            accs = np.array([r["acc"] for r in runs])
            key = f"{arch}_{var}"
            res[key] = dict(arch=arch, teacher=t, alpha=a, params=sum(q.numel() for q in model.parameters()),
                            runs=runs, acc_mean=float(accs.mean()), acc_min=float(accs.min()),
                            acc_max=float(accs.max()), auc_mean=float(np.mean([r["auc"] for r in runs])),
                            logloss_mean=float(np.mean([r["logloss"] for r in runs])))
            v = res[key]
            print(f"  {key:22s} ({v['params']:,} params) acc {v['acc_mean']:.4f} [{v['acc_min']:.4f}, "
                  f"{v['acc_max']:.4f}] auc {v['auc_mean']:.4f} logloss {v['logloss_mean']:.4f}", flush=True)
            (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["build", "train"], required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()
    build() if args.stage == "build" else train(args.threads, args.seeds, args.alpha)


if __name__ == "__main__":
    main()
