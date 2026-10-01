# A. HFT market-making latency landscape

Method note: web search plus page fetches. Several PDFs were verified by extracting text locally. Items marked **[verified]** were seen in fetched primary text. Items marked **[search snippet]** come from search-result summaries only. Reliability: H = primary or audited, M = credible secondary, L = vendor marketing, blog or anecdote.

## Summary (key takeaways)

1. **Per-firm tick-to-trade numbers are essentially not public.** I found no first-party figure from Jane Street, Virtu, Citadel Securities, HRT, Jump, IMC, Tower or XTX. Public evidence is limited to job posts showing these firms hire FPGA/ASIC engineers, a few talks, and vendor or blog claims.
2. **Latency tiers (order of magnitude).**
   - Software C++ on a tuned kernel-bypass stack: single-digit to tens of microseconds. Optiver's CppCon talk is the best public C++ source, but I could not extract exact numbers from the slides.
   - FPGA: tens to hundreds of ns. Jane Street's hardware lead says hardware latency variation is "like 20 nanoseconds across the entire range", while software is good to p99 and then goes into microseconds. Vendor tick-to-trade claims are 98-650 ns.
   - ASIC: no public numbers found.
3. **Exchange matching engines are about 10-60 µs round trip at the gateway** (Nasdaq ~14 µs door to door, SIX X-stream INET ~10 µs, Eurex T7 HF ~60 µs). Crypto venues quote sub-millisecond, and the figures are loose.
4. **Cross-venue distance dominates.** Chicago to NJ microwave is ~4.0 ms one-way, so a strategy that must react to remote signals is bounded by physics, not compute.
5. **The race is measured at the 5-10 µs scale.** The FCA/Budish study found the modal latency-arbitrage race is won by 5-10 µs. Roughly 22% of FTSE-100 volume is in races. Latency arbitrage is a ~0.5 bp tax (~$5B/yr globally).
6. **Relevance to distillation.** Audited ML inference on FPGA reaches ~1.5-2 µs p99 for small LSTMs (STAC-ML), while GPU inference is ~5 µs (GH200, best case) to tens of µs. Deep teacher models therefore sit outside the decision window. Only tiny students (small MLP/LSTM, or lookup/decision-tree logic) fit the sub-µs to few-µs budget.

## Detailed findings

### Who runs what (what is public)
- Jane Street runs a hardware team on FPGAs and builds Hardcaml. Its [Signals & Threads "Programmable Hardware" episode](https://signalsandthreads.com/programmable-hardware/) says hardware latency varies by about 20 ns, and software is good to p99 and then goes into microseconds **[verified]**. No tick-to-trade figure is given. See also [Jane Street performance engineering](https://www.janestreet.com/performance-engineering/) and the [FPGA job post](https://janestreet.com/join-jane-street/position/4279246002).
- HRT has a hardware team doing FPGA/ASIC for "low latency trading decisions" ([HRT hardware roles](https://www.hudsonrivertrading.com/hrtbeat/hrt-filter/hardware)). No latency numbers were found.
- Optiver: Carl Cook's [CppCon 2017 talk](https://isocpp.org/blog/2018/09/cppcon-2017-when-a-microsecond-is-an-eternity-high-performance-trading-syst) ([slides](https://www.smallake.kr/wp-content/uploads/2023/04/When-a-Microsecond-Is-an-Eternity-Carl-Cook-CppCon-2017.pdf)). The main point is that the hot path runs rarely and unpredictably, so cache warmth and jitter matter more than average speed. My text extraction of the slides did not yield reliable numbers, so quote the slides directly if needed.
- Citadel Securities: a QuantVPS blog says "10 microseconds" ([blog](https://www.quantvps.com/blog/top-10-high-frequency-trading-firms-dominating-global-markets)). This is unsourced and **L**.
- Virtu, Jump, IMC, Tower, XTX: nothing specific found. Generic FPGA-hiring lists mention Jump, IMC, DRW and XTX ([FPGA interview blog](https://www.techinterview.org/post/3233477296/fpga-interview-low-latency-trading-firm/)), which is low-quality secondary material.

### FPGA tick-to-trade (vendor claims)
- Exegy nxAccess claims 350 ns (pattern matcher) and 650 ns (normalized book) ([Exegy](https://www.exegy.com/nxaccess-tradingtech-insight-award-usa23/)). **L**.
- The 2017 Solarflare/Xilinx/Algo-Logic announcement quotes 98 ns, and "250 to 120 ns" in another report ([HPCwire](https://www.hpcwire.com/aiwire/2017/05/05/speed-fix-wall-street-tick-trade-half-life-cut-half/), [TE10](https://technologyevangelist.co/tag/electronic-trading/)). **L**. These definitions typically exclude the strategy logic and exchange-specific order generation, so they are lower bounds.
- Wire-to-wire "under 25 ns" is a blog claim ([blog](https://levelup.gitconnected.com/inside-high-frequency-trading-systems-the-race-to-zero-latency-faa638d0c180)) and is **L**.

### ML inference latency (directly relevant to the project)
- STAC-ML Markets (Inference) is an audited benchmark. Myrtle.ai VOLLO on FPGA reported 99p latency of about 24 µs earlier ([STAC](https://docs.stacresearch.com/node/47398)). An April 2026 press release says it reached about 2 µs, and about 1.5 µs for LSTM_A ([release](https://www.aap.com.au/aapreleases/cision20260429ae43196/)). The audit is **H**, the press wording is **L/M**.
- NVIDIA A100: LSTM_A p99 35.2 µs (1 NMI) ([STAC](https://stacresearch.com/node/47395)). GH200: 4.70 µs ([STAC SMC250910](https://docs.stacresearch.com/SMC250910)). [search snippet] for both numbers, and the models are small LSTMs, not deep teachers.

### Exchange matching-engine latency
- SIX X-stream INET (Nasdaq technology): the measured OUCH round trip averaged 10.6 µs, 95% within ~12.9 µs, 99% within ~18.4 µs ([SIX PDF](https://www.six-group.com/dam/download/the-swiss-stock-exchange/trading/trading-platform/x-stream-inet-performance-measurement-details.pdf)). I extracted these from the PDF text. The table has two columns (likely two measurement periods) that I could not label with certainty. **M-H**.
- Nasdaq: the "14 µs average, 99% in 25 µs" and "sub-40 µs" figures are **[search snippet]** from marketing-era sources. The [Nasdaq page](https://www.nasdaqtrader.com/snippets/inet2.html) describes the methodology but, when fetched, gave no numbers.
- CME Globex: an anecdotal forum claim of 15-16 µs TCP ack at a tap ([NexusFi](https://nexusfi.com/showthread.php?p=676906)) is **L**. CME states that its goal is low variability ([Waters](https://www.waterstechnology.com/trading-tech/2218958/new-cme-globex-platform-now-live-with-a-new-matching-engine-interfaces-coming)). No audited CME latency figure found.
- Eurex T7: a ~60 µs HF-session round-trip floor appears in a [Substack](https://hftadvisory.substack.com/p/venue-specific-latency-why-deterministic) (**L**). Eurex/Xetra do publish per-member Corvil stats ([Trade News](https://www.thetradenews.com/eurex-cuts-average-roundtrip-time-to-five-milliseconds)).
- Crypto: Coinbase (Deribit derivatives engine) claims "sub-millisecond" matching ([Coinbase blog](https://coinbase.com/blog/coinbase-launches-a-new-matching-engine)); the page returned 403 to me, so this is a [search snippet] and vendor-claimed (**L**). Binance "microsecond" and "0.1-1 ms" figures come from non-authoritative explainers ([Google Sites](https://sites.google.com/view/decentralised-news/home/how-crypto-exchange-matching-engines-actually-work)), which are **L**. Real-world crypto API latency is dominated by network and cloud region (milliseconds).

### Network
- McKay Brothers/Quincy microwave: Aurora to Secaucus was 8.27 ms round trip in Feb 2013 and 4.056 ms one-way by June 2015 ([Quincy](https://quincy-data.com/media/20121221-quincy-data-announces-launch-of-extreme-low-latency-wireless), [Waters](https://waterstechnology.com/node/2470547)). **M** (vendor source, but concrete and dated).
- Kernel bypass and co-location are discussed qualitatively in the Cook slides. I found no authoritative stack-latency numbers (Solarflare/Onload etc.) in this session.

### Timing of quote/cancel decisions, alpha decay, adverse selection
- Budish-Cramton-Shim (QJE 2015, [paper](https://conference.nber.org/conf_papers/f69665.pdf)): in ES-SPY, the median arbitrage duration fell from 97 ms (2005) to 7 ms (2011), while profits stayed roughly constant. The arms race raises the speed bar, not the prize. Market makers' stale quotes are "sniped", so they must cancel and reprice as quickly as possible. **H**.
- Aquilina-Budish-O'Neill ([FCA OP50](https://www.fca.org.uk/publications/occasional-papers/occasional-paper-no-50-quantifying-high-frequency-trading-arms-race-new-methodology), [NBER](https://www.nber.org/system/files/working_papers/w29011/w29011.pdf)): the modal race is won by 5-10 µs, ~537 races per FTSE-100 symbol per day, and races are ~22% of volume and about one-third of price impact and effective spread. **H**.
- IEX adds a 350 µs speed bump to give resting orders time to reprice ([Hedge Fund Journal](https://thehedgefundjournal.com/the-sec-approves-the-investors-exchange-speed-bump/)). This is a market-design response to the same picking-off problem. **M**.
- Dispute: some evidence suggests fast traders do not systematically pick off stale quotes (survey in the same search, via [TSX](https://www.tsx.com/en/resource/3111)). I did not verify it.

## Latency figures table

| Figure | Source | Reliability |
|---|---|---|
| Modal latency-arb race won by 5-10 µs | [FCA OP50](https://www.fca.org.uk/publications/occasional-papers/occasional-paper-no-50-quantifying-high-frequency-trading-arms-race-new-methodology) | H |
| ES-SPY arb duration median 97 ms (2005) to 7 ms (2011) | [BCS 2015](https://conference.nber.org/conf_papers/f69665.pdf) | H |
| FPGA hardware latency variation ~20 ns | [Jane Street podcast](https://signalsandthreads.com/programmable-hardware/) | M-H (a single quote) |
| FPGA tick-to-trade 98-650 ns | [Exegy](https://www.exegy.com/nxaccess-tradingtech-insight-award-usa23/), [HPCwire](https://www.hpcwire.com/aiwire/2017/05/05/speed-fix-wall-street-tick-trade-half-life-cut-half/) | L |
| FPGA LSTM inference p99 ~1.5-2 µs | [STAC-ML press](https://www.aap.com.au/aapreleases/cision20260429ae43196/) | L-M (audited, press-reported) |
| GH200 LSTM_A p99 4.7 µs | [STAC](https://docs.stacresearch.com/SMC250910) | M-H [snippet] |
| SIX INET OUCH round trip avg 10.6 µs, p99 ~18 µs | [SIX](https://www.six-group.com/dam/download/the-swiss-stock-exchange/trading/trading-platform/x-stream-inet-performance-measurement-details.pdf) | M-H |
| Nasdaq ~14 µs avg | search snippet, unverified | L |
| CME ~15-16 µs ack | [NexusFi](https://nexusfi.com/showthread.php?p=676906) | L |
| Eurex T7 HF ~60 µs | [Substack](https://hftadvisory.substack.com/p/venue-specific-latency-why-deterministic) | L |
| Coinbase "sub-millisecond"; Binance "microseconds" | [Coinbase](https://coinbase.com/blog/coinbase-launches-a-new-matching-engine) | L |
| Chicago-NJ microwave 4.056 ms one-way | [Quincy/McKay](https://quincy-data.com/media/20121221-quincy-data-announces-launch-of-extreme-low-latency-wireless) | M |
| IEX speed bump 350 µs | [HFJ](https://thehedgefundjournal.com/the-sec-approves-the-investors-exchange-speed-bump/) | M |
| Citadel Securities ~10 µs execution | [QuantVPS](https://www.quantvps.com/blog/top-10-high-frequency-trading-firms-dominating-global-markets) | L (unsourced) |

## Open questions
- Exact Optiver numbers from the Cook slides, and any other C++-stack latency talks (CppCon/Meeting C++ from IMC, HRT, Citadel).
- Authoritative exchange latency numbers for CME Globex and Binance/Coinbase spot (exchange-published statistics, or Corvil/Pico reports).
- ASIC tick-to-trade: no public data found. Do public ESMA/SEC documents give firm-level latency? I did not find any.
- Kernel-bypass stack latencies (Onload/DPDK/ef_vi) from vendor docs.
- Decision timing: how real market makers split latency between signal, risk checks and order encoding. The needed budget for the student model is therefore an inference, not a measured figure.
- The 5-10 µs race scale comes from LSE equities (UK). Does it hold for US futures and crypto?

## Source list
- [BCS 2015](https://conference.nber.org/conf_papers/f69665.pdf)
- [FCA OP50](https://www.fca.org.uk/publications/occasional-papers/occasional-paper-no-50-quantifying-high-frequency-trading-arms-race-new-methodology)
- [NBER w29011](https://www.nber.org/system/files/working_papers/w29011/w29011.pdf)
- [Jane Street Signals & Threads](https://signalsandthreads.com/programmable-hardware/)
- [Jane Street performance engineering](https://www.janestreet.com/performance-engineering/)
- [Jane Street FPGA job](https://janestreet.com/join-jane-street/position/4279246002)
- [HRT hardware roles](https://www.hudsonrivertrading.com/hrtbeat/hrt-filter/hardware)
- [Cook CppCon blog](https://isocpp.org/blog/2018/09/cppcon-2017-when-a-microsecond-is-an-eternity-high-performance-trading-syst)
- [Cook slides](https://www.smallake.kr/wp-content/uploads/2023/04/When-a-Microsecond-Is-an-Eternity-Carl-Cook-CppCon-2017.pdf)
- [STAC-ML VOLLO](https://docs.stacresearch.com/node/47398)
- [STAC-ML A100](https://stacresearch.com/node/47395)
- [STAC-ML GH200](https://docs.stacresearch.com/SMC250910)
- [Myrtle.ai release](https://www.aap.com.au/aapreleases/cision20260429ae43196/)
- [SIX INET performance](https://www.six-group.com/dam/download/the-swiss-stock-exchange/trading/trading-platform/x-stream-inet-performance-measurement-details.pdf)
- [Nasdaq OUCH/ITCH measurement](https://www.nasdaqtrader.com/snippets/inet2.html)
- [Waters on new CME Globex](https://www.waterstechnology.com/trading-tech/2218958/new-cme-globex-platform-now-live-with-a-new-matching-engine-interfaces-coming)
- [NexusFi CME thread](https://nexusfi.com/showthread.php?p=676906)
- [Trade News on Eurex](https://www.thetradenews.com/eurex-cuts-average-roundtrip-time-to-five-milliseconds)
- [HFT Advisory Substack](https://hftadvisory.substack.com/p/venue-specific-latency-why-deterministic)
- [Coinbase blog](https://coinbase.com/blog/coinbase-launches-a-new-matching-engine)
- [Crypto matching engines explainer](https://sites.google.com/view/decentralised-news/home/how-crypto-exchange-matching-engines-actually-work)
- [Quincy Data](https://quincy-data.com/media/20121221-quincy-data-announces-launch-of-extreme-low-latency-wireless)
- [Waters on McKay/Quincy](https://waterstechnology.com/node/2470547)
- [Hedge Fund Journal on IEX](https://thehedgefundjournal.com/the-sec-approves-the-investors-exchange-speed-bump/)
- [TSX paper](https://www.tsx.com/en/resource/3111)
- [Exegy](https://www.exegy.com/nxaccess-tradingtech-insight-award-usa23/)
- [HPCwire tick-to-trade](https://www.hpcwire.com/aiwire/2017/05/05/speed-fix-wall-street-tick-trade-half-life-cut-half/)
- [TE10](https://technologyevangelist.co/tag/electronic-trading/)
- [Gitconnected blog](https://levelup.gitconnected.com/inside-high-frequency-trading-systems-the-race-to-zero-latency-faa638d0c180)
- [techinterview FPGA blog](https://www.techinterview.org/post/3233477296/fpga-interview-low-latency-trading-firm/)
- [QuantVPS](https://www.quantvps.com/blog/top-10-high-frequency-trading-firms-dominating-global-markets)
