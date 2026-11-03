"""Stage 3b: event-time features and the next-mid-change label.

Input  $HFT_DATA_DIR/book_events/<day>/<SYM>.parquet
Output $HFT_DATA_DIR/features_event/<day>/<SYM>.parquet

Rows: best bid/ask updates (last update per timestamp), 09:35-15:55 ET, thinned to at most MAX_ROWS per
symbol-day by an even stride. Features use rows with ts <= t only; time windows use searchsorted on ts.

Label (no horizon to choose, no flat class):
    y_next = 1 if the next mid change strictly after t is up, 0 if down
    dt_next_ms = time until that change

    python code/data/features_event.py 2025-12-08 ... --symbols QQQ PYPL NVDA
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAX_ROWS = 300_000
MS = 1_000_000
EPS = 1e-9
IN_DIR = config.DATA_DIR / "book_events"
OUT_DIR = config.DATA_DIR / "features_event"


def back_idx(ts, w):
    """Index of the last row with ts <= t - w (0 if none)."""
    return np.maximum(np.searchsorted(ts, ts - w, side="right") - 1, 0)


def wsum(cs, ts, w):
    """Sum over (t - w, t] given cs = cumulative sum aligned with rows."""
    return cs - cs[back_idx(ts, w)]


def make_features(b: pl.DataFrame, sym_id: int, rng_seed=0):
    b = b.unique("ts", keep="last", maintain_order=True)
    g = {c: b[c].to_numpy().astype(np.float64) for c in b.columns}
    ts = b["ts"].to_numpy()
    bp1, ap1, bs1, as1 = g["bp1"], g["ap1"], g["bs1"], g["as1"]
    mid = (bp1 + ap1) / 2
    f = {}
    f["spread"] = ap1 - bp1
    f["qi1"] = (bs1 - as1) / (bs1 + as1 + EPS)
    for n in (3, 5):
        bb = sum(g[f"bs{i}"] for i in range(1, n + 1))
        aa = sum(g[f"as{i}"] for i in range(1, n + 1))
        f[f"di{n}"] = (bb - aa) / (bb + aa + EPS)
    f["micro_dev"] = (bp1 * as1 + ap1 * bs1) / (bs1 + as1 + EPS) - mid
    f["gap_bid"] = np.where(g["bp2"] > 0, bp1 - g["bp2"], 5.0)
    f["gap_ask"] = np.where(g["ap2"] > 0, g["ap2"] - ap1, 5.0)
    depth1 = (bs1 + as1) / 2
    depth5 = sum(g[f"bs{i}"] + g[f"as{i}"] for i in range(1, 6)) / 2
    f["l1_share"] = depth1 / (depth5 + EPS)

    # trailing 5-minute average depth (row-weighted) as the normaliser
    idx = np.arange(len(ts))
    j5 = back_idx(ts, 300_000 * MS)
    cs_d = np.cumsum(depth1)
    norm = (cs_d - cs_d[j5]) / np.maximum(idx - j5, 1) + 1.0
    norm[0] = depth1[0] + 1
    f["depth1_rel"] = np.log((depth1 + 1) / norm)

    # event-level order-flow imbalance between consecutive updates (Cont, Kukanov & Stoikov 2014)
    pb, pa = np.r_[bp1[0], bp1[:-1]], np.r_[ap1[0], ap1[:-1]]
    qb, qa = np.r_[bs1[0], bs1[:-1]], np.r_[as1[0], as1[:-1]]
    e = (np.where(bp1 >= pb, bs1, 0) - np.where(bp1 <= pb, qb, 0)
         - np.where(ap1 <= pa, as1, 0) + np.where(ap1 >= pa, qa, 0))
    e[0] = 0
    cs_e = np.cumsum(e)
    for n in (10, 100):
        f[f"ofi_ev{n}"] = (cs_e - cs_e[np.maximum(idx - n, 0)]) / norm
    for name, w in (("10ms", 10), ("100ms", 100), ("1s", 1000)):
        f[f"ofi_{name}"] = wsum(cs_e, ts, w * MS) / norm

    for name, w in (("10ms", 10), ("100ms", 100), ("1s", 1000), ("10s", 10_000), ("60s", 60_000)):
        f[f"ret_{name}"] = mid - mid[back_idx(ts, w * MS)]

    dmid = np.r_[0.0, np.diff(mid)]
    moved = dmid != 0
    last_pos = np.maximum.accumulate(np.where(moved, idx, 0))
    f["last_move"] = np.sign(dmid[last_pos])
    f["since_move"] = np.log1p((ts - ts[last_pos]) / MS)
    cs_m = np.cumsum(moved)
    f["moves_1s"] = np.log1p(wsum(cs_m, ts, 1000 * MS))
    f["moves_10s"] = np.log1p(wsum(cs_m, ts, 10_000 * MS))
    cs_abs = np.cumsum(np.abs(dmid))
    f["absmove_10s"] = wsum(cs_abs, ts, 10_000 * MS)
    f["absmove_60s"] = wsum(cs_abs, ts, 60_000 * MS)

    ones = np.cumsum(np.ones(len(ts)))
    for name, w in (("10ms", 10), ("100ms", 100), ("1s", 1000)):
        f[f"upd_{name}"] = np.log1p(wsum(ones, ts, w * MS))

    buy, sell, evn = g["cum_buy"], g["cum_sell"], g["cum_events"]
    for name, w in (("100ms", 100), ("1s", 1000), ("10s", 10_000)):
        bb, ss = wsum(buy, ts, w * MS), wsum(sell, ts, w * MS)
        f[f"tfi_{name}"] = (bb - ss) / (bb + ss + EPS)
    span5 = np.maximum((ts - ts[j5]) / (1000 * MS), 1.0)  # seconds covered by the 5-min window
    vol = buy + sell
    rate_v = (vol - vol[j5]) / span5 + EPS
    rate_e = (evn - evn[j5]) / span5 + EPS
    f["vol_rel_1s"] = np.log1p(wsum(vol, ts, 1000 * MS) / rate_v)
    f["evt_rel_1s"] = np.log1p(wsum(evn, ts, 1000 * MS) / rate_e)
    f["tod"] = (ts - config.SESSION_START_NS) / (config.SESSION_END_NS - config.SESSION_START_NS)

    # label: direction of the next mid change strictly after t (timestamps are unique after dedup)
    chg = np.nonzero(moved)[0]
    k = np.searchsorted(chg, idx, side="right")
    has = k < len(chg)
    nxt = chg[np.minimum(k, len(chg) - 1)]
    y = np.where(has, (mid[nxt] > mid).astype(np.int8), -1).astype(np.int8)
    dt = np.where(has, (ts[nxt] - ts) / MS, np.nan)

    out = {"ts": ts, "sym": np.full(len(ts), sym_id, np.int16), "mid": mid}
    out.update({k_: v.astype(np.float32) for k_, v in f.items()})
    out["y_next"] = y
    out["dt_next_ms"] = dt.astype(np.float32)
    df = pl.DataFrame(out).filter(
        (pl.col("ts") > config.SESSION_START_NS + config.TRIM_NS) &
        (pl.col("ts") <= config.SESSION_END_NS - config.TRIM_NS) & (pl.col("y_next") >= 0))
    if df.height > MAX_ROWS:
        df = df.gather_every(int(np.ceil(df.height / MAX_ROWS)))
    return df


def feature_names(df):
    skip = {"ts", "mid", "y_next", "dt_next_ms", "day"}
    return [c for c in df.columns if c not in skip]


def load(days, symbols):
    return pl.concat([pl.read_parquet(OUT_DIR / d / f"{s}.parquet").with_columns(pl.lit(d).alias("day"))
                      for d in days for s in symbols])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("days", nargs="+")
    p.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    args = p.parse_args()
    for d in args.days:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
        for s in args.symbols:
            df = make_features(pl.read_parquet(IN_DIR / d / f"{s}.parquet"), config.CANDIDATES.index(s))
            df.write_parquet(OUT_DIR / d / f"{s}.parquet")
            print(f"  {d} {s}: {df.height:,} rows, up share {df['y_next'].mean():.3f}, "
                  f"median dt_next {df['dt_next_ms'].median():.0f} ms", flush=True)


if __name__ == "__main__":
    main()
