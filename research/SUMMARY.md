# Research Summary: Distilled Models for Latency-Constrained HFT Market Making

Collated from six reports (A–F) in this folder, written 2026-10-01. Each claim points to the report that holds the links and the reliability flags. **Nothing here has been independently re-verified by the collator**; see "Verification TODO" at the end.

| Report | Topic |
|---|---|
| [A](A_hft_latency_landscape.md) | HFT latency landscape (firms, venues, alpha decay) |
| [B](B_low_latency_inference_cpp_fpga_cache.md) | C++ / FPGA / L1-cache inference engineering |
| [C](C_distillation_and_compression_methods.md) | Distillation, pruning, quantization; LLM vs traditional |
| [D](D_dl_for_lob_and_market_making.md) | DL for limit order books, critiques, candidate papers |
| [E](E_data_and_simulation.md) | L2/L3 data sources, simulators, queue and latency models |
| [F](F_llms_in_trading_and_latency.md) | LLM trading agents, LLM latency, GPU prices, distilling Ollama models |

## 1. The problem, in one paragraph

Market-making quote/cancel decisions have to land within microseconds, because stale quotes get picked off. The audited FCA/Budish study finds the modal latency-arbitrage race is won by 5–10 µs (A). Large LOB networks (CNN-LSTM, Transformers) and certainly LLMs cannot run in that window. The proposed project trains a large offline teacher, distils it into a tiny student that fits the latency budget, and measures how much of the teacher's edge survives once latency, queue position, and costs are modelled.

## 2. Key findings

### Latency reality (A, B)
- **Firm-level tick-to-trade numbers are not public.** No first-party figures from Jane Street, Virtu, Citadel Securities, HRT, Jump, IMC, Tower or XTX. The only primary-ish quote is Jane Street's hardware lead saying FPGA latency varies by about 20 ns across its whole range, while software is fine to p99 and then goes into microseconds (A, podcast). Anything else about specific firms is vendor or blog material.
- **Tiers:** software C++ on kernel bypass is single-digit to tens of µs; FPGA is tens to hundreds of ns (vendor-claimed 98–650 ns, excluding strategy logic); ASIC has no public numbers (A). Exchange gateways are about 10–60 µs round trip (SIX INET measured 10.6 µs average, p99 ~18 µs). Cross-venue distance dominates (Chicago–NJ microwave ~4 ms one-way).
- **Budget implication:** the decision path has to be a tight single-core loop. The student model is an inference engineering problem as much as a modelling one.

### Inference engineering (B)
- **FPGA:** an hls4ml MLP runs on the scale of 100 ns (arXiv 1804.06913, physics task, verified). For finance-specific audited numbers, STAC-ML LSTM_A on FPGA is single-digit to low tens of µs p99 depending on the run (see discrepancy note below).
- **GPU:** batch-1 GPU inference sits at tens of µs at best (A100 LSTM_A 35–54 µs, GH200 ~4.7 µs), so GPUs are for the teacher, not the hot path.
- **CPU and cache:** no citable ns benchmark exists for small C++ MLPs. The 0.1–0.5 µs figure for a ~40 KB int8 MLP is the agent's estimate. **We must measure this ourselves, and it is a contribution because the literature rarely reports deployment latency.**
- **L1-cache idea is sound:** L1D is 32 KB (Zen 4) to 48 KB (Golden Cove), at 4–5 cycles. That gives a budget of about **30k int8 parameters** for L1, or about 500k in a 1 MB L2 at ~14 cycles. int4 doubles capacity but needs QAT.
- **Kernel bypass** (Onload ~1–2 µs, ef_vi sub-µs) is blog or vendor level only.

### Distillation methods (C)
- **Core recipe for our student:** Hinton soft-label distillation (or regression distillation on teacher outputs) into a tiny MLP, small GRU/1D-CNN, shallow tree or GBDT, then int8 quantization. Distilling into trees or lookup tables is the extreme-latency option.
- **LLM distillation is a different regime** (token distributions, sequence-level and on-policy KD). Only the soft-target and on-policy (DAgger-style) ideas transfer to a tiny numeric student.
- **Frontier labs:** it is confirmed that Anthropic ("frontier AI labs routinely distill their own models"), Google (Gemini 1.5 Flash online-distilled from Pro; Gemma 2) and OpenAI (a distillation API) distil. **Which specific models were distilled, and whether this explains Opus 5.5's gains, is not public.** Treat the original Opus 5.5 theory as unconfirmed motivation, not a premise.
- **Known limits:** capacity gap; fidelity gap even with enough capacity (Stanton et al. 2021); little gain if the teacher isn't much better than a hard-label student; distillation scaling laws say training a teacher for just one student is often no better than supervised training (C). Trees often beat deep nets on tabular data, so a GBDT trained directly is a mandatory baseline.

### DL for LOB (D)
- **DL vs simple baselines:** evidence leans against deep stacks at short horizons. Briola 2020, TLOB 2025 and Wang 2025 find MLPs or linear models match or beat CNN-LSTM/Transformers, and Kolm et al. find that order-flow inputs matter more than architecture. Sirignano & Cont and Lucchese et al. do find real predictability from large models pooled across stocks.
- **Out of sample:** LOBCAST and TLOB report performance decay on new data. LOBFrame (Briola 2024) reports that accuracy gains do not become profit once costs apply.
- **Pitfalls:** smoothed-mean labels that leak future information, overlapping windows, FI-2010 being only 5 stocks and 10 days, and backtests that ignore queue position and latency.
- **The gap:** no paper found that distils a large LOB network into a small one. Close neighbours: Fu et al. (AAAI 2026, LLM teacher into small students for market making), Hedges 2026 (inference-compute frontier on FI-2010), distillation for volume prediction (2208.07232), and TIPS (2603.16985). C and D reached the same conclusion independently, but it rests on search coverage, not proof, so the claim needs hedging ("to our search") and a closer look at Fu et al. and Hedges.
- **Honest risk:** if simple models already match deep ones at short horizons, the "teacher edge" may be small and the finding may be "distillation closes a small gap for free" or "the teacher was never much better". D frames this as a legitimate result.

### Data and simulation (E)
- **Plan:** record Coinbase L3 yourself (free, forward-only, true FIFO queue reconstruction); use Databento's $125 free credits for a few days of ES or SPY MBO (Standard plan $199/month); use FI-2010 and LOBSTER samples for benchmarking; backtest with **hftbacktest**.
- **hftbacktest** supports separate feed and order latency, RiskAverse / probabilistic / L3-FIFO queue models, and both L2 and L3 data (verified from docs). It cannot model your own market impact, so keep order size small.
- **Method:** sweep latency (0.1 / 1 / 10 / 50 ms or the µs equivalent), add the measured compute time of teacher and student, and report markouts at several horizons, fill and cancel-before-fill rates, and PnL at several fee levels.

### LLMs (F)
- **LLM trading agents** are evaluated on daily-or-slower horizons, and nothing credible shows LLMs making or cancelling quotes at sub-second scale. StockBench finds most agents fail to beat buy-and-hold; Alpha Arena S1 was short and noisy (4 of 6 models lost money). Memorization and look-ahead contamination inflate in-window results.
- **LLMs on numeric series:** Tan et al. (NeurIPS 2024) find LLMs add nothing for time-series forecasting. No evidence found of a zero-shot LLM competitive on LOB prediction (absence of evidence, not proof).
- **Latency:** Qwen3-30B-A3B on an RTX 4090 decodes at roughly 70–175 tok/s from blog aggregates, so ~0.5–3 s per decision. That is 10^3 to 10^6 slower than an HFT loop. This makes the LLM arm a clean infeasibility demo.
- **Distilling Ollama models:** Ollama serves quantized GGUF files and is not a logit teacher. Use original Hugging Face weights with TRL `GKDTrainer` for an LM student. For our numeric student, plain soft-label distillation from our own teacher is the practical route, and the LLM stays a baseline.
- **GPU rental:** 4090 at about $0.34–0.74/hr, A100 80GB at about $1.59–2.79/hr, H100 at about $2.89–3.99/hr (RunPod and Lambda read from pricing pages; Vast.ai snippets only). A 4090 for ~20 h is under ~$20.
- **Legitimate LLM role:** slow path only (news, regime, risk parameters at seconds-to-minutes cadence).

## 3. Discrepancies between reports

- **STAC-ML FPGA latency:** A reports about 1.5–2 µs p99 (a 2026 press release, audited benchmark, press wording). B reports 5.1 µs (later suite) and about 24 µs (earlier run) for the same LSTM_A. These are probably different hardware or benchmark generations, and neither agent could open the STAC pages. Quote a range and cite the STAC page directly after checking it.
- **A100 LSTM_A p99:** 35.2 µs (A) vs 54.1 µs (B). Likely different model-instance counts (1 vs 16). Check the configuration.
- **ICC/Xelera ~1.1 µs:** vendor marketing, model unspecified. Don't cite as a benchmark.
- **Hedges 2026 (arXiv 2606.25986):** it appears in B as "FastBiNLOB" and in D as an inference-compute frontier paper. This is the same paper, and it is closest prior work to the project, so read it in full.

## 4. Implications for the project design

1. **Anchor paper (course requirement):** D's top suggestion is **DeepLOB on FI-2010** (high feasibility, standard teacher, public data). Briola 2020 (MLP vs CNN-LSTM) is the second option and directly supports the "small model suffices" premise. Pick one and state clearly what extends beyond it (distillation + latency).
2. **Student budget:** at most about 30k int8 params for L1 residency, about 500k for L2. Candidates: MLP, tiny GRU/1D-CNN, shallow tree/GBDT, lookup table.
3. **Ablations that matter** (C): student trained on hard labels only vs distilled; soft vs regression targets; extra teacher-labelled data; student family at matched latency; capacity sweep; int8 PTQ vs QAT; and a GBDT baseline.
4. **Evaluation must go past accuracy** (D, E): spread-aware labels, chronological splits, and a latency-aware replay with queue position. The headline plot is PnL (or IC) vs latency per model, with the crossover where the student overtakes the teacher.
5. **Measure our own latency** (B): a C++ harness (rdtsc, single core, batch 1) reporting p50/p99/p99.9 for each model. Do not rely on literature numbers.
6. **LLM arm** (F): present it as an infeasibility demo with and without latency applied, using a post-cutoff or anonymized sample to avoid contamination.
7. **Data licensing** (E): Databento, LOBSTER and Tardis data usually can't be redistributed. Keep raw data out of the public repo (`data/*` is already gitignored).

## 5. Verification TODO (before anything goes in the report or poster)

The collator tried to spot-check arXiv IDs via the arXiv API, but it did not respond reliably from this environment, so **no IDs were re-checked in this step**. Check by hand:

- **Cited from memory ([M] / [ID] tags):** TradingAgents 2412.20138, FinMem 2311.13743, StockAgent 2407.18957, FinGPT 2306.06031, BloombergGPT 2303.17564, FinBen 2402.12659, LLMTime 2310.07820, Time-LLM 2310.01728, Chronos 2403.07815, TimesFM 2310.10688 (all F). Most foundational IDs in C, including Hinton 1503.02531, FitNets, born-again nets, DML, Deep Compression, lottery ticket, DistilBERT, TinyBERT, MiniLLM, Gemma 2/3, DeepSeek-R1, Grinsztajn 2207.08815, and the Gemini 1.5 report 2403.05530.
- **Vendor or snippet-only numbers:** all STAC-ML figures (A, B), Nasdaq 14 µs, CME 15–16 µs, Eurex 60 µs, Coinbase/Binance "sub-millisecond", Onload/ef_vi, ICC/Xelera, Vast.ai prices.
- **Unverified claims in E:** Tardis free first-day-of-month data and plan prices, LOBSTER prices, Databento per-GB MBO rates, paper DOIs/URLs, storage estimates, fee assumptions.
- **Not retrieved:** Optiver/Cook CppCon slide numbers (A), Avellaneda-Stoikov 2008 link and LOBSTER documentation (D), DeepLOB's headline F1 numbers (D, from memory).
- **Recent, not peer reviewed:** the 2026 arXiv items (Hedges 2606.25986, LOBIN 2608.02424, T-KAN 2601.02310, TIPS 2603.16985, Fu et al. 2511.07110). T-KAN is a student journal paper.
- **Read before claiming novelty:** Fu et al. and Hedges, to confirm the "no LOB-net-to-small-net distillation" gap still holds.
