# F. LLMs in Trading and the LLM Latency Question

Verification key: [V] = fetched/read this session; [S] = seen in search snippet only; [M] = from memory, not re-verified (check before citing).

## Summary

- LLM trading agents are evaluated at daily-to-multi-day horizons on news/fundamentals. Nothing credible shows them making or cancelling quotes at millisecond or microsecond scale. The best-controlled benchmark (StockBench) finds most LLM agents fail to beat buy-and-hold. Live contests (Alpha Arena S1) were short, noisy and mostly lost money.
- Look-ahead/memorization is a first-order validity threat for any LLM backtest inside the model's training window. Zero-shot LLMs on raw numeric series add little over small purpose-built models (Tan et al., NeurIPS 2024).
- Latency: a local 30B-A3B MoE on a 4090 decodes about 70-175 tok/s (very setup-dependent). Emitting even a ~30-token decision plus prefill takes hundreds of ms to seconds, 4-6 orders of magnitude slower than an HFT quote loop (microseconds to low milliseconds). That is a clean "infeasibility" demonstration for our LLM arm.
- Distillation of an Ollama model into a tiny numeric student is possible only loosely. Ollama serves quantized GGUFs. For real logit distillation use the original HF weights with TRL GKDTrainer. But GKD is built for token-level LM students. For our task (order-book features to action) the practical path is label/soft-label distillation from our own numeric teacher into a small MLP/GRU, with the LLM only as a benchmarked baseline.

## LLM trading agent landscape

| Work | Year | Setup | Finding | Link |
|---|---|---|---|---|
| StockBench [V abstract] | 2025 | Multi-month, daily prices+fundamentals+news, buy/sell/hold; GPT-5, Claude-4, Qwen3, Kimi-K2, GLM-4.5; return, max drawdown, Sortino | Most agents struggle to beat buy-and-hold; some show better risk management; static finance-QA skill does not transfer to trading | https://arxiv.org/abs/2510.02209 |
| Alpha Arena S1 (nof1) [S] | 2025 | 6 LLMs, $10k each, Hyperliquid crypto perps, 18 Oct-3/4 Nov 2025 | Qwen3-Max ended ~$12.2k, DeepSeek ~$10.5k, others lost (GPT-5 ~$4.1k). 4 of 6 lost money. 2.5 weeks, one market regime, n=1: anecdote, not evidence | https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/amp |
| TradingAgents [M] | 2024 | Multi-agent LLM firm (analysts, debaters, trader, risk) on daily stocks | Paper reports higher cumulative return/Sharpe than baselines over a short backtest; authors' own results, short window, contamination not addressed | https://arxiv.org/abs/2412.20138 |
| FinMem [M] | 2023 | LLM agent with layered memory, single-stock daily trading | Positive backtest vs baselines; short period, memorization risk | https://arxiv.org/abs/2311.13743 |
| StockAgent [M] | 2024 | LLM agents in simulated exchange, studying external-factor effects | Simulation study, not realistic profit evidence | https://arxiv.org/abs/2407.18957 |
| FinGPT [M] | 2023 | Open-source LoRA fine-tuned LLMs for sentiment/forecast | Strong on sentiment classification; trading claims are modest | https://arxiv.org/abs/2306.06031 |
| BloombergGPT [M] | 2023 | 50B domain LLM | Better on finance NLP tasks; no trading evaluation | https://arxiv.org/abs/2303.17564 |
| FinBen [M] | 2024 | 36 datasets, 24 tasks incl. a trading task | LLMs good at extraction/sentiment, weak at forecasting and complex reasoning | https://arxiv.org/abs/2402.12659 |
| Lookahead-bias test (LAP) [S] | 2025 | Date-only recall query to measure memorization; headlines-to-returns and earnings-call tasks | LLM forecast power is inflated on high-memorization pairs and loses significance after the training cutoff | https://arxiv.org/abs/2512.23847 |
| The Memorization Problem [S] | 2025 | Contamination in financial LLM evaluation | In-sample accuracy rises with contamination (40.8% to 52.5%) while out-of-sample falls (47% to 42%) | https://arxiv.org/abs/2504.14765 |

Take-away: positive results come mostly from short, self-run backtests inside or near the training window. Controlled or live evaluations are sobering. All operate at human-ish timescales.

## LLMs on numeric time-series / order-book data

- Tan et al., "Are Language Models Actually Useful for Time Series Forecasting?" (NeurIPS 2024) [V via search]: ablating the LLM or replacing it with a simple attention layer does not hurt, often helps. Pretrained LLMs beat from-scratch models neither in accuracy nor few-shot, and cost far more compute. https://arxiv.org/abs/2406.16964
- LLMTime (Gruver et al. 2023, zero-shot by tokenizing numbers) [M] and Time-LLM (reprogramming LLMs, 2023) [M] are the papers being critiqued. https://arxiv.org/abs/2310.07820, https://arxiv.org/abs/2310.01728
- Purpose-built pretrained TS models (Chronos, TimesFM) [M] are small relative to LLMs (tens to hundreds of millions of parameters at the small end) and are the fairer "foundation model" comparison, but they target generic forecasting, not LOB microstructure at microsecond latency. https://arxiv.org/abs/2403.07815, https://arxiv.org/abs/2310.10688
- I found no peer-reviewed evidence of a zero-shot text LLM predicting limit-order-book mid-price moves competitively with DeepLOB-style models. Treat this as absence of evidence, and our experiment would be the (cheap) demonstration. Caveat: LOB data tokenized as text is very long (10 levels x 4 values x many events), which inflates prefill cost too.

## Latency / throughput numbers

Numbers are third-party blog aggregates (flag: not rigorous, batch size 1, Q4_K_M, varies with context/driver). Run our own measurement and report that.

| Model / hardware | Decode tok/s | TTFT | Source |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M, RTX 4090 24GB (llama.cpp) | ~83 | ~2.3 s (long prompt; one source) | https://willitrunai.com/es/can-run/qwen-3-30b-a3b-on-rtx-4090-24gb [S] |
| Qwen3-30B-A3B, RTX 4090, Ollama, community range | 70-176 depending on quant/context | not reported | https://markaicode.com/benchmarks/ollama-qwen-3-rtx-4090-latency-benchmark/ [V] |
| Qwen3-8B Q4_K, RTX 4090, Ollama | 141 (4K ctx) falling to 34 (128K ctx) | not reported; prefill ~65-100x faster than decode | same [V] |
| Qwen3-30B-A3B Q4_K_M, Mac Studio M3 Ultra | ~110 | n/a | https://mustafa.net/llm-tokens-per-second-benchmarks/ [S] |
| same, Mac Studio M2 Ultra | ~91 | n/a | same [S] |
| same, MacBook Pro M4 Max | ~68 | n/a | same [S] |

Back-of-envelope (ours): at 80 tok/s, a 30-token output is ~0.4 s decode, plus TTFT, so roughly 0.5-3 s per decision. HFT quote/cancel loops are microseconds to ~1 ms. Gap about 10^3 to 10^6. Even 0.5B models at several hundred tok/s remain far above 1 ms for generation. Reasoning ("thinking") models multiply output tokens by 10-100x. The Qwen3-30B-A3B Q4 footprint is ~23 GB [S], so it just fits a 24GB card; offloading to system RAM drops speed sharply. Note: Apple Silicon has high-ish bandwidth but long-prompt prefill is slow.

## GPU rental prices (on-demand, approximate; change often)

| GPU | RunPod (secure) | Lambda | Vast.ai marketplace | Source |
|---|---|---|---|---|
| RTX 4090 | $0.74/hr [V]; community ~$0.34 [S] | n/a | ~$0.34-0.50 [S] | https://www.runpod.io/pricing |
| A100 80GB | $1.59/hr [V] | $2.79/hr (8x SXM) [V]; $2.06 cited elsewhere [S] | ~$0.50-0.80 [S] | https://lambda.ai/pricing |
| H100 80GB | $2.89 PCIe / $3.49 SXM [V] | $3.99 SXM [V] | ~$0.90-1.87 [S] | https://vast.ai/blog/how-much-does-it-cost-to-rent-a-gpu-in-the-cloud |
| L40S / 5090 | $1.09 / $0.99 [V] | | | https://www.runpod.io/pricing |

Marketplace (Vast) prices vary by host and reliability. Budget: a 4090 for ~20 h of inference plus student training is under ~$20. Comparison sites (spheron, jarvislabs) are partly marketing.

## How to run the LLM arm and how to distill

LLM arm (infeasibility demo):
1. Replay a recorded LOB sample (e.g. 1,000-5,000 snapshots). Serialize top-N levels and recent trades to a short text prompt.
2. Query a local Ollama model (qwen3:0.6b/8b/30b-a3b, gemma, llama) with a constrained output (`{"action": "quote|cancel|hold"}`, temperature 0, `format` JSON, `num_predict` small, thinking off). Ollama's `/api/generate` response includes `prompt_eval_duration`, `eval_duration`, `eval_count`: use these for TTFT/throughput [M, from Ollama API docs, verify].
3. Report (a) per-decision latency p50/p99 versus our student (microseconds on CPU), (b) accuracy/PnL of LLM decisions versus teacher and student when replayed with latency-induced staleness (decision applied N ms after the snapshot). The stale-quote effect is the strongest argument, more than raw speed.
4. Control for the look-ahead issue: use data after the model's training cutoff, or anonymize tickers/dates.
5. Optional: run the larger model on a rented 4090 (~$0.34-0.74/hr) to show that more hardware does not close a 10^3+ gap.

Distillation:
- Ollama cannot be used directly as a teacher for training: GGUF quantized weights, and the API does not give full logit distributions [M, verify current logprobs support]. Take the original HF checkpoint instead.
- If the student is a small LM (e.g. Qwen3-0.6B from a Qwen3 teacher): HF TRL `GKDTrainer` (https://huggingface.co/docs/trl/gkd_trainer [V]). Key args: `lmbda` (fraction on-policy student samples), `beta` (JSD interpolation, 0 = forward KL), `seq_kd`; dataset is chat "messages". It is under `trl.experimental`, and teacher/student need compatible tokenizers. Unsloth gives memory-efficient LoRA fine-tuning [M]. llama.cpp is for exporting and running the result as GGUF, not training.
- For our actual project (tiny numeric student): treat the large model (deep LOB net, or LLM as a weak teacher) as a labeler. Train the student with soft-label KL plus hard targets (standard Hinton distillation) on features. Use the LLM only as a baseline, not as the teacher: its numeric signal is weak (see above).

## LLMs' legitimate slow-path role

News/earnings/filing sentiment, regime tags, event flags and narrative risk summaries at seconds-to-minutes cadence can set slow parameters: inventory limits, spread widening, quoting on/off kill-switch, volatility multipliers. The fast path stays a deterministic tiny model. FinGPT/BloombergGPT-style results (strong on sentiment/extraction) support this role, while contamination caveats still apply. This is a design argument, not something we need to prove empirically.

## Sources

- StockBench: https://arxiv.org/abs/2510.02209 (pdf https://arxiv.org/pdf/2510.02209)
- Alpha Arena S1 coverage: https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/amp ; https://iweaver.ai/blog/alpha-arena-ai-trading-season-1-results (marketing-flavored)
- Tan et al.: https://arxiv.org/abs/2406.16964 ; https://github.com/BennyTMT/LLMsForTimeSeries
- Lookahead bias: https://arxiv.org/abs/2512.23847 ; https://arxiv.org/abs/2504.14765
- Latency: https://willitrunai.com/es/can-run/qwen-3-30b-a3b-on-rtx-4090-24gb ; https://markaicode.com/benchmarks/ollama-qwen-3-rtx-4090-latency-benchmark/ ; https://mustafa.net/llm-tokens-per-second-benchmarks/ ; https://levelup.gitconnected.com/benchmarking-llm-inference-on-rtx-4090-rtx-5090-and-rtx-pro-6000-76b63b3b50a2
- Pricing: https://www.runpod.io/pricing ; https://lambda.ai/pricing ; https://vast.ai/blog/how-much-does-it-cost-to-rent-a-gpu-in-the-cloud ; https://www.spheron.network/blog/runpod-vs-vastai-2026/
- TRL GKD: https://huggingface.co/docs/trl/gkd_trainer ; GKD paper https://arxiv.org/abs/2306.13649
- Not re-verified [M]: TradingAgents 2412.20138, FinMem 2311.13743, StockAgent 2407.18957, FinGPT 2306.06031, BloombergGPT 2303.17564, FinBen 2402.12659, LLMTime 2310.07820, Time-LLM 2310.01728, Chronos 2403.07815, TimesFM 2310.10688 (arXiv IDs from memory; confirm before citing).
