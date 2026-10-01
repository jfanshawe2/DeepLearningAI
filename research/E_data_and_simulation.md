# Topic E: Data and Simulation Infrastructure

Date checked: 2026-10-01. Items marked **[verified]** were seen on an official page in this session. Items marked **[unverified]** come from memory or secondary sources and must be checked before relying on them. Several official pages (Databento docs, LOBSTER pricing, Tardis pricing) returned truncated content through the fetch tool, so exact prices need a manual look.

## Summary and recommended data plan

Recommended plan (cheapest path that still supports queue-aware backtests):

1. **Primary: record Coinbase Exchange (BTC-USD or ETH-USD) yourself**, using the public WebSocket feed. It is free, and Tardis docs state Coinbase can provide L3 (market-by-order) data **[verified]**. Record the full-order-level channel plus trades from a cheap VM for 3-7 days. This gives true L3, so queue position can be reconstructed exactly (FIFO), which is the best fit for a latency/queue study.
2. **Secondary / sanity check: Databento** with the **$125 free credits [verified]** for a few days of one liquid CME future (e.g. ES or MES) or a Nasdaq stock in MBO. This gives a professional-grade L3 benchmark and a second asset class. Standard plan is $199/month (includes 1 month of L2/L3 history) **[verified]**; usage-based pay-as-you-go is available **[verified]**. Per-GB MBO rates for specific datasets: unverified, check the pricing page and use the cost estimator in the Databento portal before downloading.
3. **Free benchmark/pretraining: FI-2010 and LOBSTER samples.** FI-2010 (Ntakaris et al. 2018) is the standard public Nasdaq Nordic LOB benchmark for deep-learning classification; useful for teacher/student comparisons against published numbers, but it has no order IDs/timestamps suited to latency simulation. LOBSTER publishes free sample files (message + orderbook, e.g. several tickers, up to 10 levels) **[from search snippet; verify at lobsterdata.com]**.
4. **Tardis.dev** for backfill if recording is not feasible: the first day of each month is reportedly downloadable without an API key **[unverified: found only via a search summary, not the official FAQ page, which did not mention it]**. Paid plans reported at ~$350-6,000/month **[unverified, secondary source]**; docs confirm a distinct academic subscription type and "no discounts" **[verified]**.

Split: train teacher/student on days 1-N, validate on later days, and do all latency/queue backtests on the L3 recording. Keep strictly chronological splits.

## Data source comparison

| Source | L2 / L3 | Cost | Free tier | Link |
|---|---|---|---|---|
| Coinbase Exchange WebSocket (self-record) | L2 and L3 (full order feed); market-data APIs are public, trading APIs need auth [verified] | Free + VM cost | Unlimited public feed | https://docs.cdp.coinbase.com/exchange/docs/websocket-channels |
| Binance WebSocket (self-record) | L2 diff depth only (no L3) | Free | Yes (public streams) | https://developers.binance.com/docs |
| Bitso WebSocket (self-record) | L2-style diff orders (exact channel details unverified) | Free | Yes | https://docs.bitso.com |
| Databento | L1, L2 (MBP-10), L3 (MBO); CME and US equities venues [verified] | $125 free credits; usage-based; Standard $199/mo; Plus $1,750/mo; Unlimited $4,500/mo [verified] | $125 credits | https://databento.com/pricing |
| Tardis.dev | L2 incremental + snapshots for most exchanges; L3 for Coinbase [verified] | Paid subs; academic tier exists; prices not verified | First-of-month free (unverified) | https://docs.tardis.dev |
| LOBSTER | Reconstructed Nasdaq LOB, up to 50 levels plus message file (ITCH-based, ns timestamps), 2007-present [secondary] | Pricing unverified; academic focus; some universities have accounts | Free sample files | https://lobsterdata.com |
| FI-2010 | 10-level LOB features, 5 Nordic stocks, 10 days, pre-processed | Free | Free | https://etsin.fairdata.fi (search "FI-2010"; URL unverified) |
| Kaggle / HuggingFace | Assorted crypto LOB dumps, quality varies | Free | Free | search kaggle.com / huggingface.co (specific datasets unverified) |
| University access | Check Columbia library (WRDS/TAQ is trades+quotes, not L3) | Free to students | n/a | ask library |

Note: Binance L2 cannot reveal individual order queue; queue models must then be probabilistic.

## Simulator comparison

| Tool | Latency model | Queue position | Data | Notes / link |
|---|---|---|---|---|
| hftbacktest (nkaz001) | Separate feed and order latency, custom models [verified] | RiskAverse, Prob (Power/Log), and L3 FIFO; NoPartialFill / PartialFill exchange models [verified] | L2 and L3, tick-by-tick, Numba/Rust, MIT, `pip install hftbacktest` [verified] | Best fit. https://github.com/nkaz001/hftbacktest ; docs https://hftbacktest.readthedocs.io |
| NautilusTrader | Latency model for order submission in backtest (details unverified) | Optional queue-position tracking for L3 (unverified) | Databento adapter, many venues | Production-grade, heavier. https://nautilustrader.io |
| ABIDES (ABIDES-Gym, ABIDES-Markets) | Agent-to-exchange latency models | Simulated agents, not replay | Synthetic multi-agent market | Good for impact/reactive agents, not realistic replay. https://github.com/jpmorganchase/abides-jpmc-public |
| mbt_gym | None realistic (discrete-time RL env) | Stylised Avellaneda-Stoikov-type fill model | Synthetic | Good RL prototype. https://github.com/JJJerome/mbt_gym |
| LOBSTER-based custom sims | Whatever you code | Whatever you code | LOBSTER msgs | Needs own matching engine logic |
| Databento examples | n/a | n/a | MBO books via `databento` Python lib (order book reconstruction) | Data tooling, pair with hftbacktest |

Key hftbacktest caveat from docs [verified]: replay cannot model your own market impact, so order size must be small.

## Queue and latency modelling methods

- **Queue position, L3 data:** with order IDs, simulate FIFO exactly: place your order at the tail, advance when orders ahead cancel or trade.
- **Queue position, L2 data:** you do not know where cancels occur. hftbacktest offers RiskAverse (only trades advance you; pessimistic) and probabilistic models where a cancel at quantity q behind/ahead of you is allocated by a function f(position/size) (power or log) [verified in docs]. Treat RiskAverse and a prob model as lower/upper bounds and report both.
- **Papers:**
  - Moallemi & Yuan, "A model for queue position valuation in a limit order book" (queue position has quantifiable value): search SSRN/Columbia (URL unverified).
  - Cont, Stoikov, Talreja (2010), "A stochastic model for order book dynamics", Operations Research: https://doi.org/10.1287/opre.1090.0780 (DOI from memory, unverified).
  - Huang, Lehalle, Rosenbaum (2015), "Simulating and analyzing order book data: The queue-reactive model", JASA: https://arxiv.org/abs/1312.0563 (from memory, unverified).
  - Lehalle & Mounjid, "Limit order strategic placement with adverse selection risk and the role of latency": arXiv search (unverified).
  - Latency and liquidity provision: https://arxiv.org/pdf/1511.04116 (appeared in search results, content not read).
- **Latency decomposition:** (a) feed latency = exchange timestamp to your local receive time (measure from your own recordings: local_ts minus exchange_ts); (b) order-entry latency = submit to ack at matching engine; (c) response latency = ack back to you. A cancel decision at time t only takes effect at t + decision_compute + order_latency, and the quote can be filled meanwhile (picked off). Model latency as an empirical or lognormal distribution, add compute time of teacher vs student (this is the key experimental variable), and sweep e.g. 0.1 / 1 / 10 / 50 ms.
- **Adverse selection / markouts:** for each fill compute signed mid-price change at horizons (10 ms, 100 ms, 1 s, 10 s) after the fill; compare average markout of fills against spread captured. Compare markout for fills by teacher vs student, and for stale (cancel too late) quotes.
- **Fees/rebates:** use the venue's current schedule; do not assume a rebate. Coinbase and Binance maker fees are tiered and generally positive (maker pays) at low volume; maker rebates in equities/futures need high tier status (all unverified, check fee pages). Run each backtest at 3 fee levels: maker 0 bps, +, and a rebate scenario, and report PnL before and after fees.

## Storage and cost estimates (rough, unverified; measure yourself)

- Coinbase BTC-USD full L3 feed: order-level messages in the order of several million per day, roughly 1-5 GB/day raw JSON, 0.2-0.6 GB/day compressed (gzip/parquet). 5 days: ~5-25 GB raw, ~1-3 GB compressed.
- Databento MBO, one CME E-mini future (ES) front month: several GB per day uncompressed binary during RTH (order of 5-20 GB/day; unverified). Free $125 credits could cover a few days; use the cost estimator (`metadata.get_cost` in the Python client) before pulling.
- LOBSTER sample: a few MB to tens of MB per ticker-day.
- Recording VM: a small cloud VM ~$5-20/month plus ~50 GB disk.

## Practical pitfalls

- **Clock sync:** local receive timestamps depend on NTP quality; use exchange timestamps where available and log both.
- **Dropped or gapped WebSocket messages:** use sequence numbers, snapshots on reconnect, and discard days with gaps. Coinbase full channel requires sequence checking.
- **Look-ahead in features:** compute features using only data available after feed latency; the student sees the book with a delay.
- **Optimistic fills:** simulating fills at touch price without queue position overstates PnL massively; also ignores cancel race (picked off).
- **No market impact:** replay backtests cannot model your effect on the book [verified].
- **Survivorship in time:** a few days of one instrument is one regime; report confidence intervals and block bootstrap.
- **Timestamp ties and out-of-order messages** in L3 data.
- **Licensing:** Databento/LOBSTER/Tardis data cannot normally be redistributed; keep raw data out of the public GitHub repo (the repo has `data/`; gitignore it).
- **Fetch limits:** the facts above that are marked unverified need a manual check before they go into the report.

## Sources

- Databento pricing: https://databento.com/pricing [verified]
- Databento credits FAQ: https://databento.com/docs/knowledge-base/new-users/usage-based-pricing-and-data-credits [page did not load fully]
- Tardis docs FAQ: https://docs.tardis.dev/faq/data and https://docs.tardis.dev/faq/general [verified: Coinbase L3, academic tier]
- LOBSTER: https://lobsterdata.com ; Univ. Vienna access page https://vdc.univie.ac.at/data-access/lobster [secondary]
- hftbacktest: https://github.com/nkaz001/hftbacktest ; https://hftbacktest.readthedocs.io/en/latest/order_fill.html [verified]
- Coinbase Exchange docs: https://docs.cdp.coinbase.com/exchange/docs/websocket-channels [only overview confirmed]
- Latency and liquidity provision: https://arxiv.org/pdf/1511.04116
