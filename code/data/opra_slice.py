"""Download a narrow slice of OPRA options from Databento: one expiry per day, near-the-money strikes.

For each trading day D in [start, end]:
  1. Read the Definition schema for <UNDERLYING>.OPT (parent symbology) to list live contracts.
  2. Choose the expiry:
       --mode constant : the first expiry at least --dte calendar days after D (a rolling ~1-week maturity)
       --mode follow   : the expiry --dte days after the first day in the range, kept fixed as it decays to 0DTE
  3. Keep calls and puts whose strike is within --band of the underlying's previous close.
  4. Print the cost (free API call). With --download, also pull --schema for the regular session
     into $HFT_DATA_DIR/databento/opra/<UNDERLYING>/<D>.<schema>.dbn.zst.

Run with --estimate-only first (the default). Needs DATABENTO_API_KEY in the environment.

    python code/data/opra_slice.py --underlying QQQ --start 2026-09-01 --end 2026-09-30 --dte 7 --band 0.03
"""
import argparse
import os
from pathlib import Path

import databento as db
import pandas as pd

ET = "America/New_York"


def session(day):
    open_ = pd.Timestamp(f"{day} 09:30", tz=ET)
    return open_, open_ + pd.Timedelta(hours=6, minutes=30)


def prev_closes(client, underlying, start, end):
    # Underlying daily bars from Nasdaq; cheap. Shift by one day so selection uses only prior information.
    bars = client.timeseries.get_range(
        dataset="XNAS.ITCH", schema="ohlcv-1d", symbols=underlying,
        start=(pd.Timestamp(start) - pd.Timedelta(days=10)).date().isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
    ).to_df()
    close = bars["close"]
    close.index = close.index.tz_convert(ET).date
    return close.shift(1)


def definitions(client, underlying, day):
    df = client.timeseries.get_range(
        dataset="OPRA.PILLAR", schema="definition", symbols=f"{underlying}.OPT", stype_in="parent",
        start=day.isoformat(), end=(day + pd.Timedelta(days=1)).isoformat(),
    ).to_df()
    if df.empty:
        return df
    df = df.drop_duplicates("raw_symbol", keep="last")
    df["expiry"] = pd.to_datetime(df["expiration"]).dt.tz_convert(ET).dt.date
    return df[df["instrument_class"].isin(["C", "P"])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", default="QQQ")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--dte", type=int, default=7)
    p.add_argument("--mode", choices=["constant", "follow"], default="constant")
    p.add_argument("--band", type=float, default=0.03, help="strike within +/- band of prior close")
    p.add_argument("--schema", default="cmbp-1")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()

    client = db.Historical()  # reads DATABENTO_API_KEY
    closes = prev_closes(client, args.underlying, args.start, args.end)
    out_dir = Path(os.environ.get("HFT_DATA_DIR", "~/hft_data")).expanduser() / "databento/opra" / args.underlying
    fixed_expiry, total = None, 0.0

    for day in pd.bdate_range(args.start, args.end).date:
        spot = closes.get(day)
        if spot is None or pd.isna(spot):
            continue  # holiday or missing bar
        defs = definitions(client, args.underlying, day)
        if defs.empty:
            continue

        target = day + pd.Timedelta(days=args.dte)
        if args.mode == "follow":
            if fixed_expiry is None:
                fixed_expiry = min(e for e in defs["expiry"].unique() if e >= target)
            expiry = fixed_expiry
            if day > expiry:
                break
        else:
            expiry = min(e for e in defs["expiry"].unique() if e >= target)

        chain = defs[(defs["expiry"] == expiry) & ((defs["strike_price"] / spot - 1).abs() <= args.band)]
        symbols = sorted(chain["raw_symbol"])
        start, end = session(day)
        cost = client.metadata.get_cost(
            dataset="OPRA.PILLAR", schema=args.schema, symbols=symbols, stype_in="raw_symbol",
            start=start.isoformat(), end=end.isoformat(),
        )
        total += cost
        print(f"{day} spot={spot:.2f} expiry={expiry} contracts={len(symbols)} cost=${cost:.2f}")

        if args.download:
            out_dir.mkdir(parents=True, exist_ok=True)
            client.timeseries.get_range(
                dataset="OPRA.PILLAR", schema=args.schema, symbols=symbols, stype_in="raw_symbol",
                start=start.isoformat(), end=end.isoformat(),
                path=out_dir / f"{day}.{args.schema}.dbn.zst",
            )

    print(f"total estimated cost: ${total:.2f}")


if __name__ == "__main__":
    main()
