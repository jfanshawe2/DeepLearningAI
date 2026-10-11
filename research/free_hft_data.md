# HFT Data Sources

Updated 2026-10-11.
- **[V]** = verified on the primary source.
- **[S]** = secondary source.
- **[U]** = user-reported, not yet checked.

Background detail is in [G](G_free_hft_data.md). The training, loss and simulation design that used to live here is now in [implementation_plan.md](implementation_plan.md).

---

## 0. Chosen data plan (decided 2026-10-11)

**Primary dataset: QQQ MBO (order-by-order) from Databento `XNAS.ITCH`, paid for with the free student credits.**

- **Why MBO:** it records every add, cancel, modify and execution with order IDs. From it we:
  - rebuild the full Nasdaq book;
  - insert our own hypothetical orders at exact FIFO queue positions;
  - replay the day with latency to simulate fills, cancels and pick-offs;
  - measure true markouts per fill.
- **Why not options:** OPRA has no MBO (see §5). Without queue position you can't tell whether a passive fill came from the front of the queue (mostly uninformed flow) or from a sweep through the level (often informed). So the profit side of the PnL can't be measured.
- **Buy MBO only.** MBP-1, MBP-10, TBBO, trades and BBO can all be derived from MBO.
- **Tooling:** hftbacktest ships a Databento MBO converter, so no raw-ITCH parser is needed for the primary dataset.
- **Free Nasdaq ITCH files (§2):** a secondary source. Use them for extra symbols and days, robustness tests on older regimes, and a cross-check against Databento on overlapping dates (Dec 8–12 2025 if bought).
- **Caveats for the write-up:**
  - Nasdaq venue only. Queue position on Nasdaq is exact; trades on other venues aren't visible.
  - QQQ queues at each price level are very large, so back-of-queue fills will be rare. Consider adding a few smaller Nasdaq names to get enough of our own fills.

---

## 1. Storage: keep data out of OneDrive

The repo lives in `~/Library/CloudStorage/OneDrive-Personal/...`. `data/` is gitignored, but **OneDrive still uploads everything inside the folder**, and these datasets run to tens or hundreds of GB. Keep all market data outside OneDrive and point code at it with an environment variable:

```bash
mkdir -p ~/hft_data/{itch_raw,coinbase_l3,databento,processed}
echo 'export HFT_DATA_DIR="$HOME/hft_data"' >> ~/.zshrc
```

Code reads `os.environ["HFT_DATA_DIR"]`, and `data/README.md` documents the layout. Raw vendor data must not go into the public repo, for licensing reasons.

---

## 2. Free: Nasdaq TotalView-ITCH sample files (equities, L3)

- **Link:** https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ **[V]**. Public directory, no login. Nasdaq's [ITCH FAQ](https://m.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf) refers to it as sample data **[S]**.
- **What it gives:** full-day ITCH 5.0, meaning every add, cancel, replace and execution, with order IDs and nanosecond timestamps, for every Nasdaq-traded symbol (SPY, QQQ, TSLA, NVDA and so on). That is enough for exact FIFO queue position on Nasdaq.
- **Days:** 20 full trading days, about 149 GB compressed. Re-listed 2026-10-11 **[V]**; dates inferred from filenames.

  | Period | Days | Size (gz) |
  |---|---|---|
  | Dec 8–12, 2025 (Mon–Fri, consecutive) | 5 | 51 GB |
  | Nov 28, 2025 (half-day) | 1 | 4.7 GB |
  | May 15 and May 18, 2026 (year inferred from upload date) | 2 | 29 GB |
  | Jun 12, 2026 | 1 | 17.9 GB |
  | 2019: Jan 30, Mar 27, May 30, Jul 30, Aug 30, Oct 18, Oct 30, Dec 30 | 8 | about 33 GB |
  | Jan 30, 2020 | 1 | 5.6 GB |
  | Jul 13 and Aug 13, 2021 | 2 | 10.9 GB |

  - The 2018 entries in the listing are only `.md5sum` stubs with no data.
  - `tvagg.gz` (295 MB) is not a day of ITCH data.
- **Format:** I verified the binary ITCH 5.0 framing by reading the first bytes of three files, including the recent `S*.txt.gz` files **[V]**.
- **Limits:**
  - It is the Nasdaq venue only, not the consolidated US book.
  - Hidden liquidity shows up only as `P` trade messages.
  - No licence terms are posted; use for research only.
  - hftbacktest has no raw-ITCH converter, so we need our own parser.

```bash
cd ~/hft_data/itch_raw
curl -C - -O "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S121025-v50.txt.gz"   # resumable
```

---

## 3. Free: Coinbase Exchange L3 websocket (crypto)

- **Docs:** https://docs.cdp.coinbase.com/exchange/websocket-feed/channels **[V]**.
- **Channels:** `full` (order received, open, match, change and done events) and `level3` (the same data in a compact form). Both are **"Authentication Required"**, which means a free Coinbase account and an API key **[V]**. `level2_batch` (aggregated L2) needs no auth.
- **Building the book:** take a REST L3 snapshot, buffer websocket messages, drop any with `sequence` ≤ the snapshot sequence, apply the rest, and resync on sequence gaps **[V]**.
- **It's forward-only:** you get days only once a recorder has been running. Expect very roughly 1–5 GB/day raw JSON for BTC-USD (an estimate); compress on write.

---

## 4. Other free sources (brief)

| Source | What | Status | Use |
|---|---|---|---|
| Bitstamp `live_orders` websocket | Crypto L3, public, no snapshot (use REST) | [S] | Alternative crypto L3 recorder |
| Bybit historical order book (`ob500`) | Crypto L2, 500 levels, daily files, no login | [S] ([walkthrough](https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e)) | Long L2 history for prediction |
| Binance public data | Trades, aggTrades, klines only | [V] ([README](https://github.com/binance/binance-public-data)) | Trade-flow features only |
| LOBSTER samples | Nasdaq-derived message and book files | [S] | Superseded by raw ITCH |
| FI-2010 | 10-level features, 5 stocks, 10 days | [S] | DeepLOB reproduction only |
| Tardis.dev "free first day of month" | Crypto L2/L3 | Unverified; not in Tardis FAQ | Don't rely on it |

---

## 5. Databento (primary: QQQ MBO, paid with credits)

- **Credits:**
  - The [pricing page](https://databento.com/pricing) states **$125** for new users, valid 6 months, one set per team **[V]**.
  - The **$200 student** credit is **[U]**.
- **Pricing:** **$1.20/GB for `XNAS.ITCH` MBO**, and credits **can** buy older history **[U, user confirmed in portal 2026-10-11]**. Run the cost estimator (portal, or `client.metadata.get_cost`) before each pull.
- **Plan restriction:** resolved. The pricing page mentions a Plus-plan requirement for older L3, but the user confirmed credits can buy older QQQ MBO.
- **Dataset:** `XNAS.ITCH` (Nasdaq TotalView-ITCH). Its schemas are MBO, MBP-1, MBP-10, TBBO, Trades, BBO-1s, BBO-1m, OHLCV-1s/1m/1h/1d, Definition, Statistics, Status **[U]**. We buy **MBO only**.
- **What to pull:** QQQ MBO over a run of consecutive trading days, e.g. 1–3 months, depending on cost and access. Optionally add a few smaller Nasdaq names for more of our own fills.

```python
import databento as db
client = db.Historical()  # DATABENTO_API_KEY in env
cost = client.metadata.get_cost(dataset="XNAS.ITCH", schema="mbo", symbols="QQQ",
                                start="2026-09-01", end="2026-10-01")
print(cost)
# then, once happy with the cost:
client.timeseries.get_range(dataset="XNAS.ITCH", schema="mbo", symbols="QQQ",
                            start="2026-09-01", end="2026-10-01",
                            path="~/hft_data/databento/QQQ_mbo_2026-09.dbn.zst")
```

For long ranges, use a batch job (`client.batch.submit_job`) instead of `get_range`, and split the files by day.

### Considered and rejected: OPRA.PILLAR (QQQ options)
- **Portal listing** **[U, user's portal view 2026-10-11]**:
  - 18 venues, 9,279 products, history from 2013-04-01.
  - **Schemas:** CMBP-1, CBBO-1s/1m, TCBBO, Trades, OHLCV, Statistics, Status, Definition.
- **No MBO or depth.** These are consolidated top-of-book schemas only, in line with Databento's [OPRA migration post](https://databento.com/blog/opra-migration) **[S]**.
  - Pick-off losses are still measurable: a trade through our stale quote is a near-certain fill.
  - Passive-fill profit isn't: without queue position, fill probability and markout can't be modelled together.
- `code/data/opra_slice.py` remains available. It selects one ~7-day expiry per day and near-the-money strikes, and is untested. Use it only if options come back as an extension.

---

## 6. Open checks

- The actual cost of the chosen QQQ MBO date range (`get_cost`).
- The $200 student credit terms.
- Licence terms for the emi.nasdaq.com sample files.
