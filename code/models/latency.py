"""Batch-1 inference latency of every model, as a CPU-only trading server would run it.

Deployment assumed: a software trader on a CPU server near the exchange, no GPU on the decision path
(batch-1 GPU inference is measured here and is slower) and no FPGA. Feature computation is not timed:
the student's 34 features are O(1) incremental updates per book event, taken as negligible next to
model inference (stated as an assumption).

Latency does not depend on weight values, so the torch models use random initialisation with the
real input shapes. LightGBM latency depends on the number and size of trees, so it is trained on one
event-time day with the same parameters and tree count as the 3-day model in learning_curve.py.

Each prediction is timed separately (perf_counter_ns) after warm-up; p50 / p99 / p99.9 in microseconds.

    python code/models/latency.py --family torch
    python code/models/latency.py --family lgbm     # separate process: torch and lightgbm libomp clash on macOS

Results are merged into results/latency/latency.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

OUT = config.RESULTS_DIR / "latency"
N_EVENT_FEATURES = 34   # event-time feature set (features_event.py), excluding sym
LGBM_TREES = 386        # best_iter of the 3-day event-time LightGBM (results/learning_curve/event.json)


def timeit(fn, n, warmup=200, sync=None):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        if sync:
            sync()
        t[i] = time.perf_counter_ns() - t0
    t /= 1e3
    return dict(p50=float(np.percentile(t, 50)), p99=float(np.percentile(t, 99)),
                p999=float(np.percentile(t, 99.9)), mean=float(t.mean()), n=n)


def bench_torch(n, only=None, cpu_only=False):
    import torch
    sys.path.insert(0, str(config.REPO / "deeplob_reproduction"))
    from deeplob import DeepLOB
    from models.mlp import MLP, SIZES
    from models.students import STUDENTS, K, L

    def mlp(size):
        return MLP(N_EVENT_FEATURES, SIZES[size], n_sym=3, dropout=0.0, n_out=2)

    models = {
        "mlp_small": (mlp("small"), (torch.randn(1, N_EVENT_FEATURES), torch.zeros(1, dtype=torch.long))),
        "mlp_large": (mlp("large"), (torch.randn(1, N_EVENT_FEATURES), torch.zeros(1, dtype=torch.long))),
        "deeplob": (DeepLOB(n_out=3, variant="paper"), (torch.randn(1, 1, 100, 40),)),          # ~60k, as published
        "deeplob_repo": (DeepLOB(n_out=3, variant="repo"), (torch.randn(1, 1, 100, 40),)),     # ~144k, authors' notebook
    }
    for name, cls in STUDENTS.items():  # students with a raw-book window (students.py)
        models[f"student_{name}"] = (cls(N_EVENT_FEATURES, 3), (torch.randn(1, N_EVENT_FEATURES),
                                                                torch.randn(1, K, 4 * L), torch.zeros(1, dtype=torch.long)))
    res = {}
    for name, (m, x) in models.items():
        if only and name not in only:
            continue
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:  # not under inference_mode: LSTM decomposition fails there
            m(*x)
        r = dict(params=params, mflops=fc.get_total_flops() / 1e6)

        torch.set_num_threads(1)
        with torch.inference_mode():
            r["cpu1_eager"] = timeit(lambda: m(*x), n)
            frozen = torch.jit.freeze(torch.jit.trace(m, x))
            r["cpu1_torchscript"] = timeit(lambda: frozen(*x), n)
        torch.set_num_threads(8)
        with torch.inference_mode():
            r["cpu8_eager"] = timeit(lambda: m(*x), n // 2)
        if torch.backends.mps.is_available() and not cpu_only:
            mg = m.to("mps")
            xg = tuple(t.to("mps") for t in x)
            with torch.inference_mode():
                r["mps_batch1"] = timeit(lambda: mg(*xg), n // 5, warmup=50, sync=torch.mps.synchronize)
                # Throughput view (how a per-prediction "forward time" like DeepLOB's Table III can arise):
                # one call on a batch of 32, divided by 32. Not usable live: filling the batch means waiting.
                xb = tuple(t.repeat(32, *[1] * (t.dim() - 1)) for t in xg)
                b = timeit(lambda: mg(*xb), n // 10, warmup=20, sync=torch.mps.synchronize)
                r["mps_batch32_per_sample"] = {k_: (v / 32 if k_ != "n" else v) for k_, v in b.items()}
            m.to("cpu")
        res[name] = r
        print(f"{name:10s} {params:>8,} params {r['mflops']:8.2f} MFLOPs | CPU1 eager p50 "
              f"{r['cpu1_eager']['p50']:8.1f} us | CPU1 TorchScript p50 {r['cpu1_torchscript']['p50']:8.1f} us | "
              f"CPU8 p50 {r['cpu8_eager']['p50']:8.1f} us | GPU p50 {r.get('mps_batch1', {}).get('p50', float('nan')):8.1f} us",
              flush=True)
    return res


def bench_lgbm(n):
    import lightgbm as lgb
    import polars as pl
    from data import features_event as fe

    tr = fe.load(["2025-12-10"], config.SYMBOLS)
    cols = [c for c in fe.feature_names(tr) if c != "sym"] + ["sym"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=8)
    m = lgb.train(params, lgb.Dataset(tr.select(cols).to_pandas(), tr["y_next"].to_numpy(),
                                      categorical_feature=["sym"]), LGBM_TREES)
    x = tr.select(cols).head(1).to_numpy().astype(np.float64)
    r = dict(trees=m.num_trees(), leaves_per_tree=63)
    r["cpu1_python_predict"] = timeit(lambda: m.predict(x, num_threads=1), n)
    print(f"lgbm       {m.num_trees()} trees x 63 leaves | CPU1 Booster.predict p50 "
          f"{r['cpu1_python_predict']['p50']:.1f} us (includes Python/C-API call overhead)", flush=True)
    return {"lgbm": r}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["torch", "lgbm"], required=True)
    p.add_argument("--n", type=int, default=5000, help="timed predictions per configuration")
    p.add_argument("--only", nargs="+", default=None, help="torch models to (re)measure; others keep old numbers")
    p.add_argument("--cpu-only", action="store_true", help="skip the GPU measurements")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "latency.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    new = bench_torch(args.n, args.only, args.cpu_only) if args.family == "torch" else bench_lgbm(args.n)
    for k, v in new.items():  # merge per model, so a CPU-only re-run keeps earlier GPU numbers
        res.setdefault(k, {}).update(v)
    res["_machine"] = dict(cpu=platform.processor() or platform.machine(), python=platform.python_version(),
                           note="Apple M1 Max, 64 GB, macOS; development numbers, not a Linux trading server")
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
