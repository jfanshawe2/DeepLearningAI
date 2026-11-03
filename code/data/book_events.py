"""Stage 2b: event-time book. One row every time the best bid/ask (price or size) changes.

Input  $HFT_DATA_DIR/itch_events/<day>/<SYM>.parquet
Output $HFT_DATA_DIR/book_events/<day>/<SYM>.parquet, regular session only:

    ts                       event time (ns since midnight ET); several events can share a timestamp
    bp1..bp5, bs1..bs5       bid prices (cents) / sizes after the event
    ap1..ap5, as1..as5       ask prices / sizes
    cum_buy, cum_sell        cumulative aggressor volume since the open (same signing as book.py)
    cum_trades, cum_events   cumulative trade count and order-event count since the open

The cumulative columns let features take any trailing time window with a searchsorted lookup.

    python code/data/book_events.py 2025-12-08 --symbols QQQ PYPL NVDA
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
from data.book import A, C, D, E, F, P, U, X, _add, _remove  # noqa: E402

NLEV = 5


@nb.njit(cache=True)
def build_events(ts, typ, oid, oid2, side, shares, price, printable, lo, ng, t_open, t_close):
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

    out_ts = np.empty(n, np.int64)
    out_book = np.empty((n, 4 * NLEV), np.int32)
    out_cum = np.empty((n, 4), np.int64)
    cum = np.zeros(4, np.int64)  # buy, sell, trades, events
    prev = np.array([-9, -9, -9, -9], np.int64)
    k = 0
    next_slot = 0

    for e in range(n):
        t = ts[e]
        live = t > t_open and t <= t_close
        tp = typ[e]
        if tp == A or tp == F:
            slot_of[oid[e]] = next_slot
            _add(next_slot, side[e], price[e] // 100 - lo, shares[e], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
            next_slot += 1
        elif tp == P:
            px = price[e] // 100
            if live:
                mid2 = (lo + best[0]) + (lo + best[1])
                if 2 * px > mid2:
                    cum[0] += shares[e]
                elif 2 * px < mid2:
                    cum[1] += shares[e]
                cum[2] += 1
        else:
            if oid[e] not in slot_of:
                continue
            s = slot_of[oid[e]]
            if tp == E or tp == C:
                sh = min(shares[e], o_sh[s])
                if live and (tp == E or printable[e] == 1):
                    if o_side[s] == 1:
                        cum[1] += sh
                    else:
                        cum[0] += sh
                    cum[2] += 1
                _remove(s, sh, o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
            elif tp == X:
                _remove(s, min(shares[e], o_sh[s]), o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
            elif tp == D:
                _remove(s, o_sh[s], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                del slot_of[oid[e]]
            elif tp == U:
                sd = o_side[s]
                _remove(s, o_sh[s], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                del slot_of[oid[e]]
                slot_of[oid2[e]] = next_slot
                _add(next_slot, sd, price[e] // 100 - lo, shares[e], o_side, o_px, o_sh, bid_lvl, ask_lvl, best, ng)
                next_slot += 1
        if not live:
            continue
        cum[3] += 1
        if best[0] < 0 or best[1] >= ng:
            continue
        bb, ba = best[0], best[1]
        if bb == prev[0] and ba == prev[1] and bid_lvl[bb] == prev[2] and ask_lvl[ba] == prev[3]:
            continue
        prev[0], prev[1], prev[2], prev[3] = bb, ba, bid_lvl[bb], ask_lvl[ba]
        out_ts[k] = t
        j, c = bb, 0
        while j >= 0 and c < NLEV:
            if bid_lvl[j] > 0:
                out_book[k, c] = lo + j
                out_book[k, NLEV + c] = bid_lvl[j]
                c += 1
            j -= 1
        while c < NLEV:
            out_book[k, c] = 0
            out_book[k, NLEV + c] = 0
            c += 1
        j, c = ba, 0
        while j < ng and c < NLEV:
            if ask_lvl[j] > 0:
                out_book[k, 2 * NLEV + c] = lo + j
                out_book[k, 3 * NLEV + c] = ask_lvl[j]
                c += 1
            j += 1
        while c < NLEV:
            out_book[k, 2 * NLEV + c] = 0
            out_book[k, 3 * NLEV + c] = 0
            c += 1
        out_cum[k] = cum
        k += 1
    return out_ts[:k], out_book[:k], out_cum[:k]


def build_day(day, sym):
    ev = pl.read_parquet(config.EVENTS_DIR / day / f"{sym}.parquet")
    adds = ev.filter(pl.col("type").is_in([A, F]) & (pl.col("ts") > config.SESSION_START_NS))
    ref = int(adds["price"].median() // 100)
    lo, hi = int(ref * 0.8), int(ref * 1.2) + 1
    c = {k: ev[k].to_numpy() for k in ["ts", "type", "oid", "oid2", "side", "shares", "price", "printable"]}
    t = time.time()
    ts, book, cum = build_events(c["ts"], c["type"], c["oid"], c["oid2"], c["side"].astype(np.int64),
                                 c["shares"].astype(np.int64), c["price"].astype(np.int64), c["printable"],
                                 lo, hi - lo, config.SESSION_START_NS, config.SESSION_END_NS)
    names = ([f"bp{i}" for i in range(1, NLEV + 1)] + [f"bs{i}" for i in range(1, NLEV + 1)]
             + [f"ap{i}" for i in range(1, NLEV + 1)] + [f"as{i}" for i in range(1, NLEV + 1)])
    df = pl.DataFrame({"ts": ts, **{n: book[:, j] for j, n in enumerate(names)},
                       **{n: cum[:, j] for j, n in enumerate(["cum_buy", "cum_sell", "cum_trades", "cum_events"])}})
    out = config.DATA_DIR / "book_events" / day
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{sym}.parquet")
    crossed = int((df["bp1"] >= df["ap1"]).sum())
    print(f"  {day} {sym}: {ev.height:,} events -> {df.height:,} L1 updates in {time.time() - t:.1f}s "
          f"(crossed {crossed})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("days", nargs="+")
    p.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    args = p.parse_args()
    for d in args.days:
        for s in args.symbols:
            build_day(d, s)


if __name__ == "__main__":
    main()
