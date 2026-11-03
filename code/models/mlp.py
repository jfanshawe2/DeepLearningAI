"""MLP classifiers on the same features, labels and split as baselines.py.

Two sizes, trained directly on hard labels (no distillation yet):
  small  ~4k params  (32 -> 64 -> 32 -> 3), the future student; fits in L1 cache as int8/fp16
  large  ~400k params (512 -> 512 -> 256), the future teacher

Inputs: standardised features (scaler fit on train) + a learned 4-d symbol embedding. Cross-entropy loss,
AdamW, early stopping on validation loss (Dec 11). Also reports batch-1 single-thread CPU latency.

    python code/models/mlp.py --horizons 10 50 100 --sizes small large
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.features import feature_names, load  # noqa: E402
from models.baselines import class_metrics  # noqa: E402

OUT = config.RESULTS_DIR / "nn"
SIZES = {"small": [64, 32], "large": [512, 512, 256]}


class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_sym, emb=4, dropout=0.0, n_out=3):
        super().__init__()
        self.emb = nn.Embedding(n_sym, emb)
        layers, d = [], n_in + emb
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x, s):
        return self.net(torch.cat([x, self.emb(s)], 1))


def arrays(df, cols, mean, std, sym_index, ycol):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - mean) / std
    s = np.array([sym_index[v] for v in df["sym"].to_list()], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(s), torch.from_numpy(df[ycol].to_numpy().astype(np.int64))


@torch.no_grad()
def predict(model, X, s, device, bs=65536):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(X[i:i + bs].to(device), s[i:i + bs].to(device)), 1).cpu())
    return torch.cat(out).numpy()


def eval_loss(model, X, s, y, device):
    p = predict(model, X, s, device)
    return float(-np.mean(np.log(p[np.arange(len(y)), y.numpy()] + 1e-12)))


def latency_us(model, n_in, n=3000):
    """Batch-1, single-thread CPU forward pass in PyTorch eager (an upper bound; C/ONNX will be faster)."""
    torch.set_num_threads(1)
    m = model.to("cpu").eval()
    x, s = torch.randn(1, n_in), torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(200):
            m(x, s)
        t = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter_ns()
            m(x, s)
            t[i] = time.perf_counter_ns() - t0
    torch.set_num_threads(8)
    return dict(p50=float(np.percentile(t, 50) / 1e3), p99=float(np.percentile(t, 99) / 1e3))


def train(size, tr, va, n_in, n_sym, device, epochs, lr, bs, seed=0, n_out=3, verbose=True):
    torch.manual_seed(seed)
    model = MLP(n_in, SIZES[size], n_sym, dropout=0.1 if size == "large" else 0.0, n_out=n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    Xtr, str_, ytr = tr
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, sb, yb = Xtr[idx].to(device), str_[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb, sb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        vl = eval_loss(model, *va, device)
        hist.append(dict(epoch=ep, train_loss=tot / len(perm), val_loss=vl, seconds=time.time() - t0))
        if verbose:
            print(f"    {size} ep{ep}: train {tot / len(perm):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--sizes", nargs="+", default=["small", "large"])
    p.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=4096)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tr = load(config.SPLIT["train"], args.symbols)
    va = load(config.SPLIT["val"], args.symbols)
    te = load(config.SPLIT["test"], args.symbols)
    cols = [c for c in feature_names(tr.drop("day")) if c != "sym"]
    Xraw = np.nan_to_num(tr.select(cols).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = Xraw.mean(0), Xraw.std(0) + 1e-6
    sym_index = {s: i for i, s in enumerate(sorted(tr["sym"].unique().to_list()))}
    print(f"device {device}; train {tr.height:,} / val {va.height:,} / test {te.height:,}; {len(cols)} features")

    results = {}
    for h in args.horizons:
        ycol = f"y_{h}"
        d = [x.filter(pl.col(ycol) >= 0) for x in (tr, va, te)]
        A = [arrays(x, cols, mean, std, sym_index, ycol) for x in d]
        results[str(h)] = {}
        for size in args.sizes:
            model, hist = train(size, A[0], A[1], len(cols), len(sym_index), device, args.epochs, args.lr, args.bs)
            proba = predict(model, A[2][0], A[2][1], device)
            y_te = A[2][2].numpy()
            m = class_metrics(y_te, proba.argmax(1), d[2]["sym"].to_numpy())
            m["test_logloss"] = float(-np.mean(np.log(proba[np.arange(len(y_te)), y_te] + 1e-12)))
            m["params"] = sum(q.numel() for q in model.parameters())
            m["history"] = hist
            m["latency_us"] = latency_us(model, len(cols))
            results[str(h)][size] = m
            torch.save(model.state_dict(), config.DATA_DIR / f"mlp_{size}_h{h}.pt")
            print(f"h={h} {size} ({m['params']:,} params): acc {m['acc']:.3f} F1 {m['macro_f1']:.3f} "
                  f"dir {m['dir_acc_when_both_move']:.3f} logloss {m['test_logloss']:.4f} "
                  f"latency p50 {m['latency_us']['p50']:.1f}us", flush=True)

    (OUT / "metrics.json").write_text(json.dumps({"symbols": args.symbols, "features": cols,
                                                  "results": results}, indent=1))


if __name__ == "__main__":
    main()
