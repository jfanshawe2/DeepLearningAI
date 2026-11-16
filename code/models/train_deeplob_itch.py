"""DeepLOB on the Nasdaq ITCH event-time task: does a raw-book deep model beat the feature models?

Same rows, label and split as the event-time results (results/learning_curve/event.json):
rows = best bid/ask updates in $HFT_DATA_DIR/features_event (thinned to <= 300k per stock-day),
label y_next = direction of the next mid change (down / up), train Dec 8-10, validate Dec 11 (early
stopping only), test Dec 12 (blind). Reference on these rows: LightGBM 0.663, MLP small 0.657,
logistic 0.638, queue-imbalance rule 0.626 accuracy.

Input as in the paper: the last 100 book updates x 40 values (ask price, ask size, bid price, bid size for
10 levels), taken from the full de-duplicated update stream in $HFT_DATA_DIR/book_events (not the thinned
rows). Normalisation follows the paper's LSE setup: z-score per stock using the previous day's statistics
(one mean/std shared by all price columns, one by all size columns, so price differences between levels
keep their meaning); the first day uses its own statistics. Values are clipped to +-10. Prices are stored as exact int16 cents and z-scored per batch in float32.

Deviations from the paper, for run time: batch 64 (paper 32), at most 20 epochs with patience 4 (paper 20)
on a training set ~10x FI-2010. Optimiser as in the paper: Adam, lr 0.01, epsilon 1. (With epsilon 1, Adam
steps are small, so larger batches learn much more slowly per epoch: batch 256 barely moved in a smoke test.)

Memory: full books are held in RAM at 2 bytes per value (~4 GB for all five days); windows are gathered per batch.

    python code/models/train_deeplob_itch.py                 # full run (several hours)
    python code/models/train_deeplob_itch.py --smoke         # 1 epoch on 50k rows

Outputs: results/deeplob_itch/metrics.json; weights and per-row predicted P(up) for train / val / test
(teacher outputs for distillation) in $HFT_DATA_DIR/deeplob_itch/.
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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

sys.path.insert(0, str(config.REPO / "deeplob_reproduction"))
from deeplob import DeepLOB  # noqa: E402

T = 100
NLEV = 10
BOOK_DIR = config.DATA_DIR / "book_events"
FEAT_DIR = config.DATA_DIR / "features_event"
OUT = config.RESULTS_DIR / "deeplob_itch"
STORE = config.DATA_DIR / "deeplob_itch"
COLS = [c for i in range(1, NLEV + 1) for c in (f"ap{i}", f"as{i}", f"bp{i}", f"bs{i}")]  # FI-2010 order
PRICE = np.array([c[1] == "p" for c in COLS])


def raw_book(day, sym):
    b = pl.read_parquet(BOOK_DIR / day / f"{sym}.parquet", columns=["ts"] + COLS)
    b = b.unique("ts", keep="last", maintain_order=True)  # same de-duplication as features_event.py
    return b["ts"].to_numpy(), b.select(COLS).to_numpy()  # int32 (N, 40): cents and shares


def stats(X):
    X = X[::10].astype(np.float64)  # every 10th update is plenty for a mean / std
    p, s = X[:, PRICE], X[:, ~PRICE]
    p = p[p > 0]  # empty levels are stored as price 0
    return p.mean(), p.std() + 1e-6, s.mean(), s.std() + 1e-6


def encode(X, st):
    """Compact exact storage. Prices: int16 cents relative to the stock-day median (exact ticks; float16
    z-scores would round a 1-cent step away at QQQ's ~$600). Sizes: float16 z-scores. The price z-score
    (p - mean) / std is applied in float32 per batch via the returned (a, b): z = int16 * b + a."""
    pm, ps, sm, ss = st
    ref = int(np.median(X[::10, 0]))
    n, c = len(X), int(PRICE.sum())
    Pi, Sz = np.empty((n, c), np.int16), np.empty((n, 4 * NLEV - c), np.float16)
    for i in range(0, n, 500_000):  # chunked to keep peak memory low
        ch = X[i:i + 500_000]
        Pi[i:i + 500_000] = np.clip(ch[:, PRICE] - ref, -32000, 32000)   # empty levels (price 0) clip
        Sz[i:i + 500_000] = np.clip((ch[:, ~PRICE].astype(np.float32) - sm) / ss, -10, 10)
    return Pi, Sz, (ref - pm) / ps, 1.0 / ps


class Split:
    """Concatenated normalised books for some stock-days, plus the sample rows pointing into them."""

    def __init__(self, days, prev_stats, limit=None, seed=0):
        prices, sizes, gidx, ys, syms, blks, ab, off = [], [], [], [], [], [], [], 0
        self.stats = {}
        for d in days:
            for s in config.SYMBOLS:
                ts, X = raw_book(d, s)
                st_today = stats(X)
                st = prev_stats.get(s, st_today)
                self.stats[s] = st_today
                prev_stats[s] = st_today
                f = pl.read_parquet(FEAT_DIR / d / f"{s}.parquet", columns=["ts", "y_next"])
                i = np.searchsorted(ts, f["ts"].to_numpy())
                assert (ts[i] == f["ts"].to_numpy()).all() and i.min() >= T - 1, (d, s)
                Pi, Sz, a, b = encode(X, st)
                prices.append(Pi)
                sizes.append(Sz)
                blks.append(np.full(len(i), len(ab), np.int16))
                ab.append((a, b))
                gidx.append(off + i)
                ys.append(f["y_next"].to_numpy().astype(np.int64))
                syms.append(np.full(len(i), config.CANDIDATES.index(s), np.int16))
                off += len(X)
                print(f"    {d} {s}: {len(X):,} updates, {len(i):,} rows", flush=True)
                del X
        self.P, self.S = np.concatenate(prices), np.concatenate(sizes)
        del prices, sizes
        self.ab = np.array(ab, np.float64)
        self.gidx, self.y, self.sym = np.concatenate(gidx), np.concatenate(ys), np.concatenate(syms)
        self.blk = np.concatenate(blks)
        if limit and len(self.y) > limit:
            keep = np.sort(np.random.default_rng(seed).choice(len(self.y), limit, replace=False))
            self.gidx, self.y, self.sym, self.blk = self.gidx[keep], self.y[keep], self.sym[keep], self.blk[keep]
        self.offs = np.arange(-T + 1, 1)

    def __len__(self):
        return len(self.y)

    def batch(self, idx, device):
        rows = self.gidx[idx][:, None] + self.offs[None, :]
        a, b = self.ab[self.blk[idx], 0][:, None, None], self.ab[self.blk[idx], 1][:, None, None]
        w = np.empty((len(idx), T, 4 * NLEV), np.float32)
        w[:, :, PRICE] = np.clip(self.P[rows] * b + a, -10, 10)   # exact ticks, z-scored in float
        w[:, :, ~PRICE] = self.S[rows]
        x = torch.from_numpy(w).unsqueeze(1).to(device)
        return x, torch.from_numpy(self.y[idx]).to(device)


@torch.no_grad()
def predict(model, data, device, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(data), bs):
        x, _ = data.batch(np.arange(i, min(i + bs, len(data))), device)
        out.append(torch.softmax(model(x), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def score(y, p, sym):
    pred = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), auc=roc_auc_score(y, p),
                logloss=log_loss(y, np.c_[1 - p, p].clip(1e-7, 1)),
                per_symbol_acc={config.CANDIDATES[s]: accuracy_score(y[sym == s], pred[sym == s])
                                for s in np.unique(sym)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on --limit rows per split (default 50k)")
    ap.add_argument("--limit", type=int, default=None, help="random subset of rows per split")
    args = ap.parse_args()
    if args.smoke:
        args.max_epochs = 1
    lim = args.limit or (50_000 if args.smoke else None)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    prev = {}
    print("loading train", flush=True)
    tr = Split(config.SPLIT["train"], prev, lim, args.seed)
    print("loading val", flush=True)
    va = Split(config.SPLIT["val"], prev, lim, args.seed)
    print("loading test", flush=True)
    te = Split(config.SPLIT["test"], prev, lim, args.seed)
    import resource
    print(f"peak RSS after loading {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB", flush=True)
    print(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} rows; books "
          f"{sum(d.P.nbytes + d.S.nbytes for d in (tr, va, te)) / 1e9:.1f} GB; {time.time() - t0:.0f}s", flush=True)

    model = DeepLOB(n_out=2, variant="paper").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1.0)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(args.max_epochs):
        model.train()
        t = time.time()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            x, y = tr.batch(perm[i:i + args.bs], device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        pv = predict(model, va, device)
        vacc = accuracy_score(va.y, (pv > 0.5).astype(int))
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_acc=vacc, val_auc=roc_auc_score(va.y, pv),
                         seconds=time.time() - t))
        print(f"  ep{ep:2d} train loss {tot / len(tr):.4f} val acc {vacc:.4f} auc {hist[-1]['val_auc']:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)
        if vacc > best:
            best, bad = vacc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    pt = predict(model, te, device)
    ref = json.loads((config.RESULTS_DIR / "learning_curve" / "event.json").read_text())["3"]
    res = dict(model="DeepLOB (paper size)", params=sum(q.numel() for q in model.parameters()),
               test=score(te.y, pt, te.sym), best_val_acc=best, epochs_run=len(hist),
               best_epoch=int(np.argmax([h["val_acc"] for h in hist])), n_train=len(tr), n_val=len(va),
               n_test=len(te), settings=vars(args), history=hist, minutes=(time.time() - t0) / 60,
               reference_same_rows={k: ref[k]["acc"] for k in ("lgbm", "mlp_small", "mlp_large", "logreg",
                                                               "rule_qi_sign") if k in ref})
    tag = "smoke" if args.smoke else f"seed{args.seed}"
    (OUT / ("metrics_smoke.json" if args.smoke else "metrics.json")).write_text(json.dumps(res, indent=1))
    if not args.smoke:
        torch.save(best_state, STORE / f"deeplob_itch_{tag}.pt")
        np.save(STORE / f"p_up_test_{tag}.npy", pt)
        np.save(STORE / f"p_up_val_{tag}.npy", predict(model, va, device))
        np.save(STORE / f"p_up_train_{tag}.npy", predict(model, tr, device))
    r = res["test"]
    print(f"TEST acc {r['acc']:.4f} auc {r['auc']:.4f} logloss {r['logloss']:.4f} | per symbol "
          f"{ {k: round(v, 4) for k, v in r['per_symbol_acc'].items()} } | same rows: {res['reference_same_rows']}",
          flush=True)


if __name__ == "__main__":
    main()
