"""Stage 2: rebuild the Nasdaq order book from ITCH events and sample it every 100 ms.

Input  $HFT_DATA_DIR/itch_events/<day>/<SYM>.parquet   (itch_parse.py)
Output $HFT_DATA_DIR/book/<day>/<SYM>.parquet, one row per 100 ms step k over 09:30-16:00 ET:

    ts                          step end time T_k (ns since midnight ET)
    bp1..bp10, bs1..bs10        bid prices (cents) and sizes (shares), best first; 0 if level absent
    ap1..ap10, as1..as10        ask prices / sizes
    buy_vol, sell_vol           aggressor volume in (T_{k-1}, T_k]: displayed executions (E/C) by resting
                                side, plus hidden (P) executions signed by price vs mid (Lee-Ready)
    n_trades, n_adds, n_cancels event counts in the interval
    last_px                     last trade price (cents, carried forward)

The book is the state after every event with ts <= T_k, so features at step k use only data up to T_k.

    python code/data/book.py 2025-12-08 --symbols QQQ ADBE
"""
import argparse
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl
from numba import types
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

A, F, E, C, X, D, U, P = (ord(c) for c in "AFECXDUP")


@nb.njit(cache=True)
def _remove(slot, sh, o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng):
    """Take sh shares off order slot; update level totals and best prices. best = [best_bid, best_ask]."""
    i = o_px[slot]
    o_sh[slot] -= sh
    if i < 0 or i >= ng:
        return
    if o_side[slot] == 1:
        bid_lvl[i] -= sh
        if bid_lvl[i] <= 0 and i == best[0]:
            j = i
            while j >= 0 and bid_lvl[j] <= 0:
                j -= 1
            best[0] = j
    else:
        ask_lvl[i] -= sh
        if ask_lvl[i] <= 0 and i == best[1]:
            j = i
            while j < ng and ask_lvl[j] <= 0:
                j += 1
            best[1] = j


@nb.njit(cache=True)
def _add(slot, side, i, sh, o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng):
    o_side[slot] = side
    o_px[slot] = i
    o_sh[slot] = sh
    if i < 0 or i >= ng:
        return
    if side == 1:
        bid_lvl[i] += sh
        if i > best[0]:
            best[0] = i
    else:
        ask_lvl[i] += sh
        if i < best[1]:
            best[1] = i


@nb.njit(cache=True)
def build(ts, typ, oid, oid2, side, shares, price, printable, lo, ng, t0, step, n_steps, nlev):
    n = ts.shape[0]
    n_orders = 0
    for e in range(n):
        if typ[e] == A or typ[e] == F or typ[e] == U:
            n_orders += 1
    o_side = np.zeros(n_orders + 1, np.int8)
    o_px = np.zeros(n_orders + 1, np.int64)
    o_sh = np.zeros(n_orders + 1, np.int64)
    slot_of = Dict.empty(key_type=types.int64, value_type=types.int64)
    bid_lvl = np.zeros(ng, np.int64)
    ask_lvl = np.zeros(ng, np.int64)
    best = np.array([-1, ng], np.int64)

    book = np.zeros((n_steps, 4 * nlev), np.int64)
    flow = np.zeros((n_steps, 5), np.int64)  # buy_vol, sell_vol, n_trades, n_adds, n_cancels
    last_px = np.zeros(n_steps, np.int64)
    cur_last = 0
    next_slot = 0
    k = 0
    unknown = 0

    for e in range(n + 1):
        t = ts[e] if e < n else np.int64(1) << 62
        # emit every snapshot whose time T_k precedes this event
        while k < n_steps and t > t0 + (k + 1) * step:
            j = best[0]
            c = 0
            while j >= 0 and c < nlev:
                if bid_lvl[j] > 0:
                    book[k, c] = lo + j
                    book[k, nlev + c] = bid_lvl[j]
                    c += 1
                j -= 1
            j = best[1]
            c = 0
            while j < ng and c < nlev:
                if ask_lvl[j] > 0:
                    book[k, 2 * nlev + c] = lo + j
                    book[k, 3 * nlev + c] = ask_lvl[j]
                    c += 1
                j += 1
            last_px[k] = cur_last
            k += 1
        if e == n:
            break
        in_session = t > t0
        tp = typ[e]
        if tp == A or tp == F:
            slot_of[oid[e]] = next_slot
            _add(next_slot, side[e], price[e] // 100 - lo, shares[e], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
            next_slot += 1
            if in_session and k < n_steps:
                flow[k, 3] += 1
        elif tp == P:
            px = price[e] // 100
            cur_last = px
            if in_session and k < n_steps:
                mid2 = (lo + best[0]) + (lo + best[1])
                if 2 * px > mid2:
                    flow[k, 0] += shares[e]
                elif 2 * px < mid2:
                    flow[k, 1] += shares[e]
                flow[k, 2] += 1
        else:
            if oid[e] not in slot_of:
                unknown += 1
                continue
            s = slot_of[oid[e]]
            if tp == E or tp == C:
                sh = min(shares[e], o_sh[s])
                if tp == C:
                    cur_last = price[e] // 100
                else:
                    cur_last = lo + o_px[s]
                if in_session and k < n_steps and (tp == E or printable[e] == 1):
                    if o_side[s] == 1:
                        flow[k, 1] += sh  # resting buy hit by an aggressive sell
                    else:
                        flow[k, 0] += sh
                    flow[k, 2] += 1
                _remove(s, sh, o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
            elif tp == X:
                _remove(s, min(shares[e], o_sh[s]), o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                if in_session and k < n_steps:
                    flow[k, 4] += 1
            elif tp == D:
                _remove(s, o_sh[s], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                del slot_of[oid[e]]
                if in_session and k < n_steps:
                    flow[k, 4] += 1
            elif tp == U:
                sd = o_side[s]
                _remove(s, o_sh[s], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                del slot_of[oid[e]]
                slot_of[oid2[e]] = next_slot
                _add(next_slot, sd, price[e] // 100 - lo, shares[e], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                next_slot += 1
                if in_session and k < n_steps:
                    flow[k, 3] += 1
                    flow[k, 4] += 1
    return book, flow, last_px, unknown


def build_day(day, sym, events_dir=None, out_dir=None):
    events_dir = Path(events_dir or config.EVENTS_DIR)
    ev = pl.read_parquet(events_dir / day / f"{sym}.parquet")
    adds = ev.filter(pl.col("type").is_in([A, F]) & (pl.col("ts") > config.SESSION_START_NS))
    ref = int(adds["price"].median() // 100)
    lo, hi = int(ref * 0.8), int(ref * 1.2) + 1  # cents; orders outside +/-20% are ignored for levels
    n_steps = (config.SESSION_END_NS - config.SESSION_START_NS) // config.STEP_NS
    cols = [ev[c].to_numpy() for c in ["ts", "type", "oid", "oid2", "side", "shares", "price", "printable"]]
    cols = [cols[0], cols[1], cols[2], cols[3], cols[4].astype(np.int64), cols[5].astype(np.int64),
            cols[6].astype(np.int64), cols[7]]
    t = time.time()
    book, flow, last_px, unknown = build(*cols, lo, hi - lo, config.SESSION_START_NS, config.STEP_NS,
                                         n_steps, config.N_LEVELS)
    L = config.N_LEVELS
    names = ([f"bp{i}" for i in range(1, L + 1)] + [f"bs{i}" for i in range(1, L + 1)]
             + [f"ap{i}" for i in range(1, L + 1)] + [f"as{i}" for i in range(1, L + 1)])
    df = pl.DataFrame({n: book[:, j].astype(np.int32) for j, n in enumerate(names)})
    df = df.with_columns(
        pl.Series("ts", config.SESSION_START_NS + config.STEP_NS * np.arange(1, n_steps + 1)),
        *[pl.Series(n, flow[:, j]) for j, n in enumerate(["buy_vol", "sell_vol", "n_trades", "n_adds", "n_cancels"])],
        pl.Series("last_px", last_px.astype(np.int32)),
    ).select(["ts"] + names + ["buy_vol", "sell_vol", "n_trades", "n_adds", "n_cancels", "last_px"])
    out = Path(out_dir or config.BOOK_DIR) / day
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{sym}.parquet")
    crossed = int(((df["bp1"] >= df["ap1"]) & (df["ap1"] > 0)).sum())
    print(f"  {day} {sym}: {ev.height:,} events -> {n_steps:,} snapshots in {time.time() - t:.1f}s "
          f"(unmatched refs {unknown}, crossed snapshots {crossed})", flush=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("days", nargs="+")
    p.add_argument("--symbols", nargs="+", default=config.CANDIDATES)
    args = p.parse_args()
    for day in args.days:
        for s in args.symbols:
            build_day(day, s)


if __name__ == "__main__":
    main()
