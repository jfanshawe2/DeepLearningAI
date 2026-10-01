# Topic D: Deep Learning for LOB Prediction and Market Making, and the Problem We Pose

Every link below was returned by search or fetched during this research; arXiv abstracts were opened for the 2024-2026 items. Numeric results are quoted from abstracts or summaries unless marked "from memory". Check any figure against the paper before putting it in the report.

## 1. Summary

- Deep nets (DeepLOB and descendants) report strong mid-price-trend F1 on FI-2010 and some other datasets. Newer work finds that the gains shrink or vanish out of sample, under realistic labels, and once costs and execution are included.
- Several independent critiques find that simple models (MLPs, logistic regression, order-flow-imbalance features) match or beat the heavy CNN-LSTM/Transformer stacks. Briola et al. 2020, TLOB 2025 and Wang 2025 all say this.
- Latency is acknowledged as a constraint but rarely modelled. We found only a few papers that treat accuracy against inference cost for LOB: Hedges 2026 (inference-compute frontier), LOBIN 2026 (in-switch inference), and T-KAN 2026 (FPGA-oriented). We found no paper that does teacher-to-student distillation on LOB data. Distillation has been applied to market making with an LLM teacher (Fu et al., AAAI 2026) and to volume prediction (2208.07232). So the niche exists, but the field is filling in fast.
- The student-vs-teacher framing is well motivated. If a tiny MLP or GBDT is already near the large model, the interesting questions are how much of the teacher's edge a student can retain and whether that edge survives costs.

## 2. Problem framing

A market maker quotes bid and ask, earns the spread, and carries inventory and adverse-selection risk. Decisions (post, cancel, requote) must happen within the order-book update interval. Industry tick-to-trade budgets are microseconds on FPGAs. One source notes a couple of hundred microseconds for an unoptimised DeepLOB forward pass (a summary of the Hedges/LOBIN literature, not verified in the paper text). A large teacher (CNN-Inception-LSTM, Transformer, ensemble) is therefore too slow for the quote/cancel loop, and a stale signal increases adverse selection. Relaver (2025) shows that even modelled 30-100 ms latency creates unintended cancellations and inventory risk for RL market makers.

Our proposed problem: given a teacher trained offline with no latency limit, train a student with a fixed latency or FLOP budget (for example a small MLP, a tiny GRU, or a small tree ensemble). Evaluate along three axes: (a) classification or regression accuracy, (b) agreement with the teacher (calibration, soft-label fidelity), (c) a latency-aware trading or market-making metric that includes spread costs and signal delay.

## 3. Key papers

| Paper | Year | Model | Data | Reported result | Link |
|---|---|---|---|---|---|
| Ntakaris et al., Benchmark Dataset for Mid-Price Forecasting of LOB Data with ML Methods | 2017 | Ridge, SLFN baselines | FI-2010: 5 Helsinki (NASDAQ Nordic) stocks, 10 days | First public LOB benchmark | [arXiv:1705.03233](https://arxiv.org/abs/1705.03233) |
| Cont, Kukanov, Stoikov, The Price Impact of Order Book Events | 2014 (arXiv 2010) | Linear OFI regression | NYSE TAQ, 50 US stocks | Short-horizon price change is mainly driven by order flow imbalance; linear relation, slope inversely proportional to depth | [arXiv:1011.6402](https://arxiv.org/abs/1011.6402) |
| Sirignano, Cont, Universal Features of Price Formation | 2019 (arXiv 2018) | Large deep nets | Billions of US equity quotes and trades | Universal model beats stock-specific linear and nonlinear models; generalises to unseen stocks | [arXiv:1803.06917](https://arxiv.org/abs/1803.06917) |
| Zhang, Zohren, Roberts, DeepLOB | 2019 | CNN + Inception + LSTM | FI-2010 and London Stock Exchange | Reported state of the art at the time; LSE results claimed to generalise (details from memory) | [arXiv:1808.03668](https://arxiv.org/abs/1808.03668) |
| Zhang, Zohren, Roberts, BDLOB | 2018 | Dropout-variational Bayesian CNN | LSE, millions of observations | Posterior uncertainty used for position sizing; Bayesian dropout acts as a regulariser | [arXiv:1811.10041](https://arxiv.org/abs/1811.10041) |
| Wallbridge, TransLOB | 2020 | Causal conv + masked self-attention | FI-2010 | Claimed new state of the art on FI-2010 | [arXiv:2003.00130](https://arxiv.org/abs/2003.00130) |
| Briola, Turiel, Aste, Deep Learning Modeling of LOB: a Comparative Perspective | 2020 | Logistic regression, MLP, LSTM, attention LSTM, CNN-LSTM | Same features and data for all | MLP matches or beats CNN-LSTM; the "spatial and temporal" structure is only an approximation | [arXiv:2007.07319](https://arxiv.org/abs/2007.07319) |
| Kolm, Turiel, Westray, Deep Order Flow Imbalance | 2023 (Math. Finance) | Off-the-shelf nets (LSTM etc.) on order-flow inputs | 115 NASDAQ stocks | Stationary order-flow inputs outperform raw LOB states; effective horizon is about two price changes | [SSRN 3900141](https://papers.ssrn.com/abstract=3900141) |
| Lucchese, Pakkanen, Veraart, Short-Term Predictability of Returns in Order Book Markets | 2022 | DL with new volume representation | Multi-stock order book data | Predictability "ubiquitous"; representation matters; model confidence sets | [arXiv:2211.13777](https://arxiv.org/abs/2211.13777) |
| Prata et al., LOB-Based DL Models for Stock Price Trend Prediction (LOBCAST) | 2023 | 15 DL models | FI-2010, NASDAQ-style data | All models drop significantly on new data | [arXiv:2308.01915](https://arxiv.org/abs/2308.01915) |
| Briola, Bartolucci, Aste, Deep Limit Order Book Forecasting (LOBFrame) | 2024 | Several DL models | NASDAQ LOB | Forecast accuracy does not translate into profitable trades; microstructure drives efficacy | [arXiv:2403.09267](https://arxiv.org/abs/2403.09267) |
| Berti, Kasneci, TLOB | 2025 | Dual-attention Transformer; MLP baseline | FI-2010, NASDAQ, Bitcoin | Predictability declined over time (-6.68 F1); a simple MLP is competitive; proposes a label fix for "horizon bias"; performance degrades with spread-based labels | [arXiv:2502.15757](https://arxiv.org/abs/2502.15757) |
| Wang, Exploring Microstructural Dynamics in Crypto LOBs | 2025 | Logistic regression, DeepLOB, Conv1D+LSTM | Bybit BTC/USDT | Better inputs matter more than depth; simpler models match or exceed with faster inference | [arXiv:2506.05764](https://arxiv.org/abs/2506.05764) |
| Makinde, T-KAN | 2026 | Temporal KAN | LOB data | 19.1% relative F1 gain at k=100; claims FPGA-friendly. Student journal, so weigh the evidence accordingly | [arXiv:2601.02310](https://arxiv.org/abs/2601.02310) |
| Hedges, Inference-Compute Frontier and a Latency-Efficient Architecture for LOB Prediction | 2026 | Trees, HGB, CatBoost, MLPLOB, FastBiNLOB | FI-2010 | Power-law loss-vs-compute frontier (R^2 0.941); hardware-aware mixer hits targets at lower latency; no distillation | [arXiv:2606.25986](https://arxiv.org/abs/2606.25986) |
| Hong et al., LOBIN (in-network inference) | 2026 | ML inference inside programmable switches | NASDAQ-style feeds | Over 10% lower latency than the exchange-server benchmark; about 3% error-rate increase in the hybrid setting | [arXiv:2608.02424](https://arxiv.org/abs/2608.02424) |
| Fu et al., Distilling LLM Features Into Small Models (Cooperative Market Making) | 2026 (AAAI) | LLM teacher, multiple students, Hajek-MoE | Four real market datasets | Beats distillation and RL baselines; motivated by LLM inference speed. LLM teacher, not LOB net | [arXiv:2511.07110](https://arxiv.org/abs/2511.07110) |
| Spooner, Fearnley, Savani, Koukorinis, Market Making via RL | 2018 | TD RL, tile coding | Simulated LOB | Beats benchmark and online-learning baselines | [arXiv:1804.04216](https://arxiv.org/abs/1804.04216) |
| Spooner, Savani, Robust Market Making via Adversarial RL | 2020 | Adversarial RL on Avellaneda-Stoikov game | Simulation | Risk-averse behaviour emerges; more robust | [arXiv:2003.01820](https://arxiv.org/abs/2003.01820) |
| Gasperov et al., RL Approaches to Optimal Market Making (survey) | 2021 | Survey | n/a | RL tends to beat analytical strategies on risk-adjusted return (surveyed claims) | [MDPI](https://www.mdpi.com/2227-7390/9/21/2689/) |
| Relaver, Resolving Latency and Inventory Risk in MM with RL | 2025 | RL with hold-time state and trend predictor | Four real datasets, simulated 30-100 ms latency | Outperforms RL baselines | [arXiv:2505.12465](https://arxiv.org/abs/2505.12465) |
| Stoikov, The Micro-Price | 2018 | Closed-form Markov estimator | Intraday data | Better short-term predictor than mid or weighted mid | [Quant. Finance 18(12)](https://ideas.repec.org/a/taf/quantf/v18y2018i12p1959-1966.html) |
| Avellaneda, Stoikov, High-frequency trading in a limit order book | 2008 | Analytical inventory-skewing quotes | Theory | Standard market-making baseline (no arXiv link; cited from memory) | n/a |

Distillation outside LOB: [arXiv:2208.07232](https://arxiv.org/abs/2208.07232) does distribution-aware distillation for volume prediction, motivated by HFT latency.

## 4. Critiques and pitfalls

1. **DL vs simple baselines.** Evidence is mixed but leans against deep stacks at short horizons. Briola 2020, TLOB 2025 and Wang 2025 find MLPs or linear models competitive. Kolm et al. find that input choice (order flow) matters more than architecture. Sirignano and Cont, and Lucchese et al., find real predictability and benefit from large models and from pooling across stocks. Caveat: these results use different datasets, horizons and labels. We could not find a single paper that tests DL against microprice or OFI-linear baselines at the same horizon with costs.
2. **Out-of-sample decay.** LOBCAST reports significant degradation on new data. TLOB reports declining predictability over time.
3. **Accuracy is not PnL.** Briola 2024 and TLOB both show that F1 or accuracy gains do not become profit once spreads and fees are applied. Trend labels that ignore spread overstate usefulness.
4. **Labels.** Smoothed-mean-price labels with thresholds (FI-2010 style) use future information, depend on horizon and threshold, and are heavily class-imbalanced. TLOB claims "horizon bias" in prior labelling. Smoothed labels are hard to trade on.
5. **Leakage and splits.** Overlapping windows across train/test, normalisation fit on the whole sample, and random splits all inflate scores. FI-2010 is only 10 days and 5 stocks, so it is easy to overfit.
6. **Non-stationarity.** Raw prices and sizes are non-stationary; OFI-style inputs help (Kolm et al.).
7. **Backtest realism.** No queue position, no latency, no market impact, filling at mid, and ignoring adverse selection. These problems are acute for market making, which depends on fills and queue priority.
8. **RL market making.** Most deep RL MM results are in simulators with simplified fill models. Compare to a tuned Avellaneda-Stoikov baseline, not a naive one. Gasperov's survey reports RL beating analytical baselines, but the simulator gap is a standing concern.

## 5. Industry practice (partly from general knowledge, not verified)

Practitioners are widely reported to use linear and regularised models on engineered features (OFI, queue imbalance, microprice, trade signs, volatility), GBDTs, and small nets. The reasons are latency, robustness, and interpretability. Inference is deployed on FPGAs or tuned C++. I did not find an authoritative citable source for this paragraph, so treat it as a hypothesis in the report, supported indirectly by Stoikov 2018, Cont et al. 2014, Hedges 2026 and Wang 2025.

## 6. Gaps our project could fill

1. LOB-specific teacher-to-student distillation with a latency budget (no direct paper found). Also test which student classes (MLP, tiny GRU, trees, linear) retain the most.
2. A latency-aware evaluation: add signal delay of k events or microseconds and measure how teacher advantage decays. A slow teacher with delayed predictions may lose to a fast student.
3. A cost-aware comparison on the same data against OFI-linear and microprice baselines, with spread-aware labels.
4. A distillation target that matches trading use: regression on future mid or microprice change, or soft probabilities for position sizing, rather than hard trend labels.
5. Optionally, a simple market-making simulator (Avellaneda-Stoikov quotes skewed by the model signal) to show downstream PnL and inventory risk.

## 7. Candidate papers to reproduce

1. **DeepLOB (Zhang et al. 2019) on FI-2010.**
   - Why: the field's standard teacher; well known; public data; our natural teacher model.
   - Feasibility: high. FI-2010 is small and DeepLOB trains in minutes to an hour on a GPU. Several open implementations exist. Expect to match F1 only approximately, and the original numbers (from memory) should be checked. Risk: FI-2010 is a toy benchmark.
2. **Briola et al. 2020, comparative perspective (MLP vs CNN-LSTM vs logistic regression).**
   - Why: reproduces the claim that simple models rival deep ones, which is the core of our student premise.
   - Feasibility: high to medium. The models are small; data may need LOBSTER (paid or sample data) or FI-2010 as a substitute. We would need to state any substitution.
3. **LOBCAST (Prata et al. 2023), a subset of models and the generalisation-drop result.**
   - Why: gives a ready framework with many models (possible teachers and students), a profit analysis, and a robustness finding to reproduce.
   - Feasibility: medium. 15 models and large data are too much, so reproduce 3-4 models and the out-of-sample drop on a smaller dataset. Check the repo and licence first.
   - Alternates if these are rejected: TransLOB (heavier, less reproducible reported gains) or Kolm et al. (needs the big NASDAQ data, so low feasibility).

## 8. Sources

- Ntakaris et al. 2017: https://arxiv.org/abs/1705.03233
- Cont, Kukanov, Stoikov: https://arxiv.org/abs/1011.6402
- Sirignano, Cont: https://arxiv.org/abs/1803.06917
- DeepLOB: https://arxiv.org/abs/1808.03668
- BDLOB: https://arxiv.org/abs/1811.10041
- TransLOB: https://arxiv.org/abs/2003.00130
- Briola, Turiel, Aste: https://arxiv.org/abs/2007.07319
- Kolm, Turiel, Westray: https://papers.ssrn.com/abstract=3900141
- Lucchese et al.: https://arxiv.org/abs/2211.13777
- LOBCAST (Prata et al.): https://arxiv.org/abs/2308.01915
- Briola, Bartolucci, Aste: https://arxiv.org/abs/2403.09267
- TLOB: https://arxiv.org/abs/2502.15757
- Wang 2025: https://arxiv.org/abs/2506.05764
- T-KAN: https://arxiv.org/abs/2601.02310
- Hedges: https://arxiv.org/abs/2606.25986
- LOBIN: https://arxiv.org/abs/2608.02424
- Fu et al.: https://arxiv.org/abs/2511.07110
- Volume-prediction distillation: https://arxiv.org/abs/2208.07232
- Spooner et al. 2018: https://arxiv.org/abs/1804.04216
- Spooner, Savani 2020: https://arxiv.org/abs/2003.01820
- Gasperov et al. survey: https://www.mdpi.com/2227-7390/9/21/2689/
- Relaver: https://arxiv.org/abs/2505.12465
- Stoikov micro-price: https://ideas.repec.org/a/taf/quantf/v18y2018i12p1959-1966.html

Not verified: LOBSTER dataset documentation (lobsterdata.com, not fetched), and Avellaneda-Stoikov 2008 (no link retrieved). The 2026 arXiv IDs were opened and exist, but they are recent and not peer reviewed.
