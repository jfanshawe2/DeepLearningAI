# G. Free High-Frequency Data for Testing the Thesis

Checked 2026-10-10. **[V]** = verified on the primary source this session (page fetched, directory listed, or file bytes inspected). **[S]** = secondary source or search summary only. This report supersedes parts of [E](E_data_and_simulation.md); see "Corrections" below.

## What the thesis needs

| Gate | Question | Data requirement |
|---|---|---|
| 1 | Does a large DL model beat GBDT / MLP / linear baselines out of sample? | Lots of data: many days, ideally many instruments (DL wins in the literature mostly when pooled across stocks). L2 depth (MBP-10) is enough. |
| 2 | Does the distilled student beat same-size direct students and GBDT on latency-adjusted PnL? | Order-level (L3 / MBO) data for exact FIFO queue position, plus enough days for out-of-sample tests. |

## Headline finding: Nasdaq hosts free full-day TotalView-ITCH files

Nasdaq's public directory **https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/** serves full-day TotalView-ITCH 5.0 files with no login **[V: directory listed, HTTP 200]**. ITCH is true L3: every add, cancel, delete, replace and execution, with order reference numbers and ns timestamps, for every symbol traded on Nasdaq. Nasdaq's own [ITCH FAQ](https://m.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf) points to this directory as sample data **[S]**.

Files listed on 2026-10-10 (compressed sizes; dates inferred from filenames, MMDDYY):

| File | Inferred date | Size (gz) |
|---|---|---|
| `S120825-v50.txt.gz` | 2025-12-08 (Mon) | 8.8 GB |
| `S120925-v50.txt.gz` | 2025-12-09 (Tue) | 7.9 GB |
| `S121025-v50.txt.gz` | 2025-12-10 (Wed) | 11.6 GB |
| `S121125-v50.txt.gz` | 2025-12-11 (Thu) | 10.5 GB |
| `S121225-v50.txt.gz` | 2025-12-12 (Fri) | 12.6 GB |
| `S112825-v50.txt.gz` | 2025-11-28 (half-day, day after Thanksgiving) | 4.7 GB |
| `S061226-v50.txt.gz` | 2026-06-12 (probably) | 17.9 GB |
| `itch50_05_15.gz`, `itch50_05_18.gz` | likely 2026-05-15 / 05-18 (uploaded June 2026; year not in name) | 13.0 / 16.2 GB |
| `S071321`, `S081321`, `S101819` | 2021-07-13, 2021-08-13, 2019-10-18 | 3.9–6.0 GB |
| `MMDDYYYY.NASDAQ_ITCH50.gz` | 8 dates in 2019–2020 | 3.5–5.6 GB |

- **Format verified:** I range-downloaded the first 200 KB of `S121025-v50.txt.gz`, `01302020.NASDAQ_ITCH50.gz` and `itch50_05_15.gz`. All three decompress to the same ITCH 5.0 binary framing (2-byte length prefix, then `S` system-event and `R` stock-directory messages). Despite the `.txt` name, the recent files are binary ITCH 5.0 **[V]**.
- **Key asset:** **Dec 8–12, 2025 is a full consecutive trading week** (~51 GB compressed). That supports a clean chronological split, for example train Mon–Wed, validate Thu, test Fri, plus scattered extra days for robustness.
- **Coverage caveat:** This is the Nasdaq venue's book only, not the consolidated US market. Queue position is exact for orders resting on Nasdaq, and that is the realistic setting for a Nasdaq market maker. State it in the write-up.
- **Licensing:** The directory carries no usage terms that I saw. Use it for research and do not redistribute raw files in the public repo.
- **Paid equivalents, for reference:** Nasdaq sells the full archive ([Historical TotalView-ITCH](https://m.nasdaqtrader.com/Trader.aspx?id=ITCH) **[S]**), and Databento sells it normalised as MBO from May 2018 **[S]**.

## Other free sources

| Source | Depth | Free? | Verified status | Use for |
|---|---|---|---|---|
| **Nasdaq ITCH samples** (above) | L3 | Yes, download | [V] | Both gates. Primary dataset. |
| **Coinbase Exchange `full` / `level3` websocket** | L3 | Feed is free, but **requires authentication** (API key from a free account) | [V] [Coinbase docs](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels). `level2_batch` (L2) needs no auth | Live crypto L3 recording, forward-only |
| **Bitstamp `live_orders` websocket** | L3 (order created/changed/deleted) | Public, no auth | [S] Official page didn't render. [Tardis docs](https://docs.tardis.dev/historical-data-details/bitstamp) say there is no initial snapshot, so fetch one by REST | Alternative crypto L3 recorder; thinner volume than Coinbase |
| **Bybit historical order book** | L2, 500 levels, snapshot plus delta (`ob500`) | Free downloads, no registration, daily files | [S] [Medium walkthrough](https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e); the official page was not checked | Gate 1 in crypto: months of deep L2 for free. No queue IDs |
| **Binance public data** (data.binance.vision) | Trades, aggTrades, klines | Free | [V] [README](https://github.com/binance/binance-public-data) lists only these; no order-book files listed | Trade-flow features only; not enough alone |
| **LOBSTER samples** | L3-derived message plus book files, up to 50 levels | Free sample files | [S] Built from Nasdaq ITCH samples; tickers and dates unconfirmed | Quick prototyping; superseded by raw ITCH above |
| **FI-2010** | 10-level features, 5 Nordic stocks, 10 days | Free | [S] | Literature comparability (DeepLOB numbers) only. Too small and pre-processed to test the thesis |
| **Tardis.dev free first-of-month** | L2/L3 incl. Coinbase L3 | Possibly | [S] Only a 2022 Deribit blog says so; Tardis's FAQ doesn't mention it | Unconfirmed; don't plan around it |
| **Kaggle crypto LOB** | 1 s / 1 min / 5 min snapshots | Free | [S] | Useless for HFT; too coarse |
| **Databento** | All schemas incl. MBO | $125 free credits, expires in 6 months | [V] earlier | Optional second asset class (CME futures) |

## Corrections to earlier reports

- **E said Coinbase L3 is public.** It is free, but the `full` and `level3` channels are now marked "Authentication Required" [V]. You need a (free) Coinbase account and API key.
- **E and SUMMARY treated LOBSTER or Databento as the main equity L3 route.** The free Nasdaq ITCH files make paid equity L3 unnecessary for this project.
- **Tardis free first-day data remains unverified** and is not in Tardis's own FAQ.

## Recommended data plan

1. **Primary: Nasdaq ITCH, Dec 8–12 2025.**
   - Gate 1: rebuild MBP-10 books for a pooled universe (for example 20–100 liquid Nasdaq stocks) and train the teacher and baselines. Pooling across stocks is the condition under which the literature finds DL helps, so it gives gate 1 its fairest chance.
   - Gate 2: use the same days at L3 for queue-aware, latency-swept backtests on a handful of names.
   - Robustness: use the 2026 files (May, June) and older days as out-of-regime tests.
2. **Crypto arm (optional):** use Bybit `ob500` daily files for a long L2 history for gate 1. Run a Coinbase L3 recorder (API key) or a Bitstamp `live_orders` recorder from now on for a crypto gate-2 check.
3. **FI-2010:** only to reproduce the anchor paper's numbers (DeepLOB) for the course requirement.
4. **Databento credits:** keep in reserve. Optionally use a few days of CME ES MBO if you want a futures result.

## Practical issues

- **Storage location:** The repo lives in **OneDrive**. `data/` is gitignored, but OneDrive will still try to sync ~50 GB+ of ITCH files. Store raw data outside the OneDrive folder (for example `~/data/itch/`) and symlink or configure paths.
- **Streaming:** Uncompressed, each day is roughly 2–3× its gz size. Parse it streaming (gzip, then ITCH messages) and immediately filter to your symbol set. Don't decompress whole days to disk.
- **Parser:** There is no hftbacktest converter for raw ITCH; it only ships a converter for Databento MBO files [S, hftbacktest docs](https://hftbacktest.readthedocs.io/en/latest/reference/hftbacktest.data.utils.databento.html). Writing an ITCH 5.0 parser and book builder is an engineering task. The spec is public and open-source parsers exist (not evaluated here).
  - The book builder must apply Nasdaq semantics: Replace (`U`) cancels the order and re-adds it under a new reference number, losing priority. Executions (`E`, `C`) and cancels (`X`, `D`) reduce queue size ahead of you.
  - Output: hftbacktest's L3 event format, or our own simulator.
- **Hidden liquidity:** Hidden and non-displayed executions appear as `P` (trade) messages with no visible order. Account for them in fill modelling.
- **Download bandwidth:** ~51 GB for the week. Resume-capable downloads (`curl -C -`) and checksums are advisable; some older files ship `.md5sum` files.

## Sources

- Nasdaq ITCH directory: https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/ [V]
- Nasdaq ITCH FAQ: https://m.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf [S]
- Nasdaq Historical TotalView-ITCH: https://m.nasdaqtrader.com/Trader.aspx?id=ITCH [S]
- Coinbase Exchange websocket channels: https://docs.cdp.coinbase.com/exchange/websocket-feed/channels [V]
- Tardis Bitstamp notes: https://docs.tardis.dev/historical-data-details/bitstamp [S]
- Tardis FAQ (no mention of free data): https://docs.tardis.dev/faq/data [V]
- Deribit/Tardis 2022 post on free first-day data: https://insights.deribit.com/?p=7828 [S]
- Bybit free order book walkthrough: https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e [S]
- Binance public data: https://github.com/binance/binance-public-data [V]
- Kaggle crypto LOB: https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data [S]
- hftbacktest Databento converter: https://hftbacktest.readthedocs.io/en/latest/reference/hftbacktest.data.utils.databento.html [S]
- hftbacktest L3 tutorial: https://hftbacktest.readthedocs.io/en/latest/tutorials/Level-3%20Backtesting.html [S]
- Databento pricing: https://databento.com/pricing [V]
