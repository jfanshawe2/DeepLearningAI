"""Reproduce DeepLOB on FI-2010, Setup 2 (paper Table II): train days 1-7, test days 8-10, k = 10, 20, 50.

Data: the authors' preprocessed FI-2010 files (no auction, decimal-precision normalisation) in
$HFT_DATA_DIR/fi2010 (see data/README.md). Each file is 149 rows x N events: rows 1-40 are the
10-level book (ask price, ask size, bid price, bid size per level), rows 145-149 the labels for
k = 10, 20, 30, 50, 100 (1 = up, 2 = stationary, 3 = down; FI-2010's own smoothed labels, paper eq. 3).
Note: the paper used the z-score files; it reports only a "subtle difference" between normalisations.

Validation: the paper early-stops on validation accuracy but does not say where the FI-2010 validation
set comes from. Following the authors' released code, the last 20% of the training file (end of day 7)
is held out for early stopping. Days 8-10 are never used for any choice: the test set is blind.

Training as in the paper (Sec. V-A): Adam, lr 0.01, epsilon 1, batch 32, cross-entropy, stop when
validation accuracy has not improved for 20 epochs; the best-validation weights are tested.
Input windows are the last T = 100 events. As in the authors' code, windows are built over the
concatenated file and may span a stock or day boundary (a few hundred of ~250k windows).

Metrics: accuracy and precision / recall / F1 both weighted-averaged and macro-averaged. The paper's
Table II has recall == accuracy for every row, which holds for the weighted average, so the weighted
numbers are the ones comparable with the paper.

    python deeplob_reproduction/train_fi2010.py --k 10
    python deeplob_reproduction/train_fi2010.py --k 20 --max-epochs 3     # quick smoke test

Data directory: --data-dir, else $FI2010_DIR, else $HFT_DATA_DIR/fi2010 (default /Volumes/SeagateData/hft_data/fi2010).
Outputs: deeplob_reproduction/results/deeplob_k<k>.json (metrics + per-epoch history); weights in the data dir.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deeplob import DeepLOB  # noqa: E402

FI_DIR = Path(os.environ.get("FI2010_DIR") or
              Path(os.environ.get("HFT_DATA_DIR", "/Volumes/SeagateData/hft_data")) / "fi2010").expanduser()
OUT = HERE / "results"
LABEL_ROW = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}
PAPER_TABLE2 = {10: dict(acc=84.47, precision=84.00, recall=84.47, f1=83.40),
                20: dict(acc=74.85, precision=74.06, recall=74.85, f1=72.82),
                50: dict(acc=80.51, precision=80.38, recall=80.51, f1=80.35)}
T = 100


def load_file(name):
    a = np.loadtxt(FI_DIR / name, dtype=np.float32)   # (149, N)
    return a[:40].T.copy(), a[-5:].T.astype(np.int64) - 1   # features (N, 40); labels (N, 5) in {0, 1, 2}


class Windows:
    """Window i = rows [i-T+1, i] of X, label y[i]; gathered on the device per batch (no 4 GB copy)."""

    def __init__(self, X, y, device):
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.ends = torch.arange(T - 1, len(X), device=device)
        self.offs = torch.arange(-T + 1, 1, device=device)

    def __len__(self):
        return len(self.ends)

    def batch(self, idx):
        e = self.ends[idx]
        return self.X[e[:, None] + self.offs[None, :]].unsqueeze(1), self.y[e]


@torch.no_grad()
def predict(model, data, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(data), bs):
        xb, _ = data.batch(torch.arange(i, min(i + bs, len(data)), device=data.X.device))
        out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy(), data.y[data.ends].cpu().numpy()


def metrics(pred, y):
    r = dict(acc=100 * accuracy_score(y, pred))
    for avg in ("weighted", "macro"):
        p, rc, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        r[avg] = dict(precision=100 * p, recall=100 * rc, f1=100 * f)
    r["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, choices=[10, 20, 30, 50, 100])
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--variant", default="paper", choices=["paper", "repo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", default="DecPre", choices=["DecPre", "ZScore", "MinMax"],
                   help="FI-2010 normalisation; the paper used ZScore (files named *_ZScore_CF_*)")
    p.add_argument("--data-dir", default=None, help="folder with the FI-2010 .txt files")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (keep low to leave the machine usable)")
    args = p.parse_args()
    global FI_DIR
    if args.data_dir:
        FI_DIR = Path(args.data_dir).expanduser()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    Xtr_all, ytr_all = load_file(f"Train_Dst_NoAuction_{args.norm}_CF_7.txt")
    tests = [load_file(f"Test_Dst_NoAuction_{args.norm}_CF_{i}.txt") for i in (7, 8, 9)]
    Xte = np.concatenate([x for x, _ in tests])
    yte = np.concatenate([y for _, y in tests])[:, LABEL_ROW[args.k]]
    cut = int(len(Xtr_all) * 0.8)
    col = LABEL_ROW[args.k]
    tr = Windows(Xtr_all[:cut], ytr_all[:cut, col], device)
    va = Windows(Xtr_all[cut:], ytr_all[cut:, col], device)
    te = Windows(Xte, yte, device)
    print(f"k={args.k} device={device}: train {len(tr):,} / val {len(va):,} / test {len(te):,} windows "
          f"(loaded in {time.time() - t0:.0f}s)", flush=True)

    model = DeepLOB(n_out=3, variant=args.variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1.0)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(args.max_epochs):
        model.train()
        t = time.time()
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            xb, yb = tr.batch(perm[i:i + args.bs])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        vp, vy = predict(model, va)
        vacc = 100 * accuracy_score(vy, vp)
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_acc=vacc, seconds=time.time() - t))
        print(f"  ep{ep:3d} train loss {tot / len(tr):.4f} val acc {vacc:.2f} ({time.time() - t:.0f}s)", flush=True)
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    tp, ty = predict(model, te)
    res = dict(k=args.k, variant=args.variant, seed=args.seed, params=sum(q.numel() for q in model.parameters()),
               best_epoch=int(np.argmax([h["val_acc"] for h in hist])), epochs_run=len(hist),
               best_val_acc=best_acc, test=metrics(tp, ty), paper=PAPER_TABLE2.get(args.k),
               test_class_balance=(np.bincount(ty, minlength=3) / len(ty)).tolist(),
               settings=vars(args), history=hist, minutes=(time.time() - t0) / 60)
    tag = (f"deeplob_k{args.k}" + ("" if args.variant == "paper" else f"_{args.variant}")
           + ("" if args.norm == "DecPre" else f"_{args.norm.lower()}") + ("" if args.seed == 0 else f"_seed{args.seed}"))
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    torch.save(best_state, FI_DIR / f"{tag}.pt")
    w = res["test"]["weighted"]
    print(f"k={args.k} TEST acc {res['test']['acc']:.2f} P {w['precision']:.2f} R {w['recall']:.2f} "
          f"F1 {w['f1']:.2f} (macro F1 {res['test']['macro']['f1']:.2f}) | paper {res['paper']}", flush=True)


if __name__ == "__main__":
    main()
