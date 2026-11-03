"""Pick the mid-liquidity stocks that go alongside QQQ, using one day of rebuilt books.

For each candidate: price, mean spread (ticks), share of time at a 1-tick spread, median best-level queue
(shares), median trade size, queue length measured in trades, and trades per day. QQQ queues are huge, so
a 1-lot order at the back rarely fills; we want liquid names (enough trades and fills) whose best-level
queue is short relative to trading activity.

Rule: keep names with >= 20k trades/day whose Nasdaq spread is one tick at least half the time (otherwise
Nasdaq's displayed book is not representative of the market: for many mid-caps the Nasdaq-only spread is
10+ cents while the consolidated market is 1-2 cents); among those take the two with the fewest trades
needed to clear the best-level queue (queue / average trade size).

    python code/data/select_symbols.py 2025-12-08
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def stats(day, sym):
    b = pl.read_parquet(config.BOOK_DIR / day / f"{sym}.parquet")
    b = b.filter((pl.col("ts") > config.SESSION_START_NS + config.TRIM_NS) &
                 (pl.col("ts") <= config.SESSION_END_NS - config.TRIM_NS) &
                 (pl.col("bp1") > 0) & (pl.col("ap1") > pl.col("bp1")))
    vol = b["buy_vol"] + b["sell_vol"]
    trades = int(b["n_trades"].sum())
    spread = b["ap1"] - b["bp1"]
    queue = ((b["bs1"] + b["as1"]) / 2).median()
    trade_size = vol.sum() / max(trades, 1)
    return dict(symbol=sym, price=float(((b["bp1"] + b["ap1"]) / 200).median()),
                spread_ticks=float(spread.mean()), one_tick=float((spread == 1).mean()),
                l1_queue=float(queue), avg_trade=float(trade_size), queue_in_trades=float(queue / trade_size),
                trades=trades, volume=int(vol.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("day")
    p.add_argument("--symbols", nargs="+", default=config.CANDIDATES)
    args = p.parse_args()
    df = pl.DataFrame([stats(args.day, s) for s in args.symbols]).sort("queue_in_trades")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=2):
        print(df)
    ok = df.filter((pl.col("symbol") != "QQQ") & (pl.col("trades") >= 20_000) & (pl.col("one_tick") >= 0.5))
    pick = ok["symbol"].head(2).to_list()
    print("selected:", ["QQQ"] + pick)
    out = config.RESULTS_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / f"symbol_selection_{args.day}.csv")


if __name__ == "__main__":
    main()
