"""Stage 3: turn 100 ms book snapshots into model inputs and labels.

Input  $HFT_DATA_DIR/book/<day>/<SYM>.parquet
Output $HFT_DATA_DIR/features/<day>/<SYM>.parquet

Every feature at step k uses only snapshots <= k (rolling windows are trailing). All features are
scale-free (ticks, ratios, or values relative to that stock's own trailing average), so one model can be
pooled across stocks. Rows cover 09:35-15:55 ET; the first 5 minutes double as warm-up for the 5-minute
trailing windows.

Labels, for horizons h (steps of 100 ms), with no smoothing (DeepLOB-style smoothed labels leak future info):
    dm_h = mid[k+h] - mid[k]   (ticks)
    y_h  = 0 down / 1 flat / 2 up

    python code/data/features.py 2025-12-08 --symbols QQQ ADBE COST
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

HORIZONS = [10, 50, 100]          # 1 s, 5 s, 10 s
RET_WINDOWS = [1, 10, 50, 300, 3000]
FLOW_WINDOWS = [1, 10, 50, 300]
NORM_WINDOW = 3000                # 5 minutes, trailing normaliser
EPS = 1e-9


def trailing_sum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    out = c[w:] - c[:-w]
    return np.concatenate([c[1:w], out])  # partial sums during warm-up


def trailing_mean(x, w):
    n = np.minimum(np.arange(1, len(x) + 1), w)
    return trailing_sum(x, w) / n


def lag(x, w):
    out = np.empty_like(x)
    out[:w] = x[0]
    out[w:] = x[:-w]
    return out


def make_features(book: pl.DataFrame, sym_id: int):
    g = {c: book[c].to_numpy().astype(np.float64) for c in book.columns}
    L = config.N_LEVELS
    bp1, ap1, bs1, as1 = g["bp1"], g["ap1"], g["bs1"], g["as1"]
    valid = (bp1 > 0) & (ap1 > 0) & (ap1 > bp1)

    mid = np.where(valid, (bp1 + ap1) / 2, np.nan)
    mid = pl.Series(mid).fill_null(strategy="forward").fill_nan(None).fill_null(strategy="forward") \
        .fill_null(strategy="backward").to_numpy()
    f = {}
    f["spread"] = np.where(valid, ap1 - bp1, np.nan)
    f["qi1"] = (bs1 - as1) / (bs1 + as1 + EPS)
    for n_lv in (3, 5, 10):
        b = sum(g[f"bs{i}"] for i in range(1, n_lv + 1))
        a = sum(g[f"as{i}"] for i in range(1, n_lv + 1))
        f[f"di{n_lv}"] = (b - a) / (b + a + EPS)
    micro = (bp1 * as1 + ap1 * bs1) / (bs1 + as1 + EPS)
    f["micro_dev"] = np.where(valid, micro - mid, 0.0)
    f["gap_bid"] = np.where(g["bp2"] > 0, bp1 - g["bp2"], np.nan)
    f["gap_ask"] = np.where(g["ap2"] > 0, g["ap2"] - ap1, np.nan)

    depth1 = (bs1 + as1) / 2
    norm_depth = trailing_mean(depth1, NORM_WINDOW) + EPS
    f["depth1_rel"] = np.log((depth1 + 1) / norm_depth)
    depth10 = sum(g[f"bs{i}"] + g[f"as{i}"] for i in range(1, L + 1)) / 2
    f["depth10_rel"] = np.log((depth10 + 1) / (trailing_mean(depth10, NORM_WINDOW) + EPS))
    f["l1_share"] = depth1 / (depth10 + EPS)

    # Order-flow imbalance (Cont, Kukanov & Stoikov 2014) on consecutive snapshots
    pb, pa, qb, qa = lag(bp1, 1), lag(ap1, 1), lag(bs1, 1), lag(as1, 1)
    e = (np.where(bp1 >= pb, bs1, 0) - np.where(bp1 <= pb, qb, 0)
         - np.where(ap1 <= pa, as1, 0) + np.where(ap1 >= pa, qa, 0))
    e[0] = 0
    for w in FLOW_WINDOWS:
        f[f"ofi_{w}"] = trailing_sum(e, w) / norm_depth

    for w in RET_WINDOWS:
        f[f"ret_{w}"] = mid - lag(mid, w)
    dmid = np.abs(np.diff(mid, prepend=mid[0]))
    f["absmove_300"] = trailing_sum(dmid, 300)
    f["absmove_3000"] = trailing_sum(dmid, 3000) / 10

    buy, sell = g["buy_vol"], g["sell_vol"]
    vol = buy + sell
    norm_vol = trailing_mean(vol, NORM_WINDOW) + EPS
    for w in [10, 50, 300]:
        b, s = trailing_sum(buy, w), trailing_sum(sell, w)
        f[f"tfi_{w}"] = (b - s) / (b + s + EPS)
        f[f"vol_rel_{w}"] = np.log1p(trailing_sum(vol, w) / (norm_vol * w))
    act = g["n_adds"] + g["n_cancels"]
    f["act_rel_10"] = np.log1p(trailing_sum(act, 10) / ((trailing_mean(act, NORM_WINDOW) + EPS) * 10))
    f["last_vs_mid"] = np.where(g["last_px"] > 0, g["last_px"] - mid, 0.0)

    ts = g["ts"]
    f["tod"] = (ts - config.SESSION_START_NS) / (config.SESSION_END_NS - config.SESSION_START_NS)

    out = {"ts": ts.astype(np.int64), "sym": np.full(len(ts), sym_id, np.int16), "mid": mid}
    out.update({k: v.astype(np.float32) for k, v in f.items()})
    for h in HORIZONS:
        fut = np.concatenate([mid[h:], np.full(h, np.nan)])
        dm = fut - mid
        out[f"dm_{h}"] = dm.astype(np.float32)
        out[f"y_{h}"] = np.where(np.isnan(dm), -1, np.sign(dm) + 1).astype(np.int8)

    df = pl.DataFrame(out)
    keep = (pl.col("ts") > config.SESSION_START_NS + config.TRIM_NS) & \
           (pl.col("ts") <= config.SESSION_END_NS - config.TRIM_NS)
    return df.filter(keep)


def feature_names(df):
    skip = {"ts", "mid"} | {c for c in df.columns if c.startswith(("dm_", "y_"))}
    return [c for c in df.columns if c not in skip]


def load(days, symbols, feat_dir=None):
    feat_dir = Path(feat_dir or config.FEAT_DIR)
    parts = []
    for d in days:
        for s in symbols:
            parts.append(pl.read_parquet(feat_dir / d / f"{s}.parquet").with_columns(pl.lit(d).alias("day")))
    return pl.concat(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("days", nargs="+")
    p.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    args = p.parse_args()
    for d in args.days:
        out = config.FEAT_DIR / d
        out.mkdir(parents=True, exist_ok=True)
        for s in args.symbols:
            book = pl.read_parquet(config.BOOK_DIR / d / f"{s}.parquet")
            df = make_features(book, config.CANDIDATES.index(s))
            df.write_parquet(out / f"{s}.parquet")
            print(f"  {d} {s}: {df.height:,} rows, {len(feature_names(df))} features", flush=True)


if __name__ == "__main__":
    main()
