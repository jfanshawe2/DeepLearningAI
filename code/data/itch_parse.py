"""Stage 1: stream a Nasdaq TotalView-ITCH 5.0 day and keep the order events for chosen symbols.

The gzip file is decompressed by a `gzip -dc` subprocess (so decompression and parsing run in parallel)
and parsed chunk by chunk in numba. Output, one file per symbol:

    $HFT_DATA_DIR/itch_events/<day>/<SYM>.parquet
    columns: ts (ns since midnight ET), type (ASCII code), oid, oid2, side (+1 buy / -1 sell / 0 unknown),
             shares, price (1e-4 dollars), printable (C messages only)

Types kept: A/F add, E execute, C execute-with-price, X partial cancel, D delete, U replace,
P non-displayed trade. Side/price for E/C/X/D/U are resolved later from the order book (book.py).

    python code/data/itch_parse.py 2025-12-08 --symbols QQQ ADBE COST
Spec: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CHUNK = 128 * 1024 * 1024

# Message lengths (excluding the 2-byte length prefix), used only as a sanity check.
EVENT_TYPES = b"AFECXDUP"


@nb.njit(inline="always")
def u16(b, i):
    return (np.int64(b[i]) << 8) | np.int64(b[i + 1])


@nb.njit(inline="always")
def u32(b, i):
    return (np.int64(b[i]) << 24) | (np.int64(b[i + 1]) << 16) | (np.int64(b[i + 2]) << 8) | np.int64(b[i + 3])


@nb.njit(inline="always")
def u48(b, i):
    return (u16(b, i) << 32) | u32(b, i + 2)


@nb.njit(inline="always")
def u64(b, i):
    return (u32(b, i) << 32) | u32(b, i + 4)


@nb.njit(cache=True)
def parse_chunk(buf, n, locate_map, targets, out_ts, out_sym, out_type, out_oid, out_oid2,
                out_side, out_shares, out_price, out_print):
    """Parse complete messages in buf[:n]. Returns (bytes consumed, records written, messages seen)."""
    pos = 0
    k = 0
    msgs = 0
    ntar = targets.shape[0]
    while pos + 2 <= n:
        ln = u16(buf, pos)
        if pos + 2 + ln > n:
            break
        m = pos + 2
        t = buf[m]
        msgs += 1
        if t == 82:  # 'R' stock directory: map stock_locate -> target index
            loc = u16(buf, m + 1)
            for j in range(ntar):
                same = True
                for c in range(8):
                    if buf[m + 11 + c] != targets[j, c]:
                        same = False
                        break
                if same:
                    locate_map[loc] = j
        elif t == 65 or t == 70 or t == 69 or t == 67 or t == 88 or t == 68 or t == 85 or t == 80:
            loc = u16(buf, m + 1)
            s = locate_map[loc]
            if s >= 0:
                out_ts[k] = u48(buf, m + 5)
                out_sym[k] = s
                out_type[k] = t
                out_oid[k] = u64(buf, m + 11)
                out_oid2[k] = 0
                out_side[k] = 0
                out_shares[k] = 0
                out_price[k] = 0
                out_print[k] = 0
                if t == 65 or t == 70 or t == 80:  # A, F, P
                    out_side[k] = 1 if buf[m + 19] == 66 else -1
                    out_shares[k] = u32(buf, m + 20)
                    out_price[k] = u32(buf, m + 32)
                elif t == 69:  # E
                    out_shares[k] = u32(buf, m + 19)
                elif t == 67:  # C
                    out_shares[k] = u32(buf, m + 19)
                    out_print[k] = 1 if buf[m + 31] == 89 else 0
                    out_price[k] = u32(buf, m + 32)
                elif t == 88:  # X
                    out_shares[k] = u32(buf, m + 19)
                elif t == 85:  # U
                    out_oid2[k] = u64(buf, m + 19)
                    out_shares[k] = u32(buf, m + 27)
                    out_price[k] = u32(buf, m + 31)
                k += 1
        pos = m + ln
    return pos, k, msgs


def parse_day(path, symbols, max_bytes=None):
    targets = np.zeros((len(symbols), 8), dtype=np.uint8)
    for j, s in enumerate(symbols):
        targets[j] = np.frombuffer(s.ljust(8).encode(), dtype=np.uint8)
    locate_map = np.full(65536, -1, dtype=np.int64)

    cap = CHUNK // 12 + 1
    cols = dict(ts=np.int64, sym=np.int16, type=np.uint8, oid=np.int64, oid2=np.int64,
                side=np.int8, shares=np.int32, price=np.int32, printable=np.int8)
    bufs = {c: np.empty(cap, dtype=d) for c, d in cols.items()}
    pieces = {c: [] for c in cols}

    proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=0)
    buf = np.empty(CHUNK + 65536, dtype=np.uint8)
    have, total_in, total_msgs, t0 = 0, 0, 0, time.time()
    view = memoryview(buf)
    while True:
        # fill the buffer in place; pipe reads return at most ~64 KB at a time
        eof = False
        while have < CHUNK:
            got = proc.stdout.readinto(view[have:CHUNK])
            if not got:
                eof = True
                break
            have += got
            total_in += got
        used, k, msgs = parse_chunk(buf, have, locate_map, targets, bufs["ts"], bufs["sym"], bufs["type"],
                                    bufs["oid"], bufs["oid2"], bufs["side"], bufs["shares"],
                                    bufs["price"], bufs["printable"])
        total_msgs += msgs
        for c in cols:
            pieces[c].append(bufs[c][:k].copy())
        buf[:have - used] = buf[used:have]
        have -= used
        if total_in and total_in % (CHUNK * 16) < CHUNK:
            print(f"  {total_in / 1e9:.1f} GB, {total_msgs / 1e6:.0f}M msgs, {time.time() - t0:.0f}s", flush=True)
        if eof or (max_bytes and total_in >= max_bytes):
            break
    proc.kill()
    proc.wait()
    out = {c: np.concatenate(v) for c, v in pieces.items()}
    print(f"  done: {total_in / 1e9:.1f} GB, {total_msgs / 1e6:.0f}M msgs, kept {len(out['ts']):,}, "
          f"{time.time() - t0:.0f}s; leftover bytes {have}", flush=True)
    return out, locate_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("day", help="YYYY-MM-DD")
    p.add_argument("--symbols", nargs="+", default=config.CANDIDATES)
    p.add_argument("--file", help="override input path (e.g. a truncated sample)")
    p.add_argument("--max-bytes", type=float, default=None, help="stop after this many decompressed bytes")
    p.add_argument("--out", default=str(config.EVENTS_DIR))
    args = p.parse_args()

    path = Path(args.file) if args.file else config.itch_file(args.day)
    out, _ = parse_day(path, args.symbols, int(args.max_bytes) if args.max_bytes else None)
    out_dir = Path(args.out) / args.day
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(out)
    for j, s in enumerate(args.symbols):
        part = df.filter(pl.col("sym") == j).drop("sym")
        part.write_parquet(out_dir / f"{s}.parquet")
        print(f"  {s}: {part.height:,} events")


if __name__ == "__main__":
    main()
