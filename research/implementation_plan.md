# Implementation Plan: Distilled Low-Latency Prediction for Market Making

Rewritten 2026-11-03 (scope cut for the Dec 8 presentation). The previous, larger design (own replay engine,
counterfactual bandit labels, RL) is in git history (commit 94060ac) and is now out of scope. Data sources are
in [free_hft_data.md](free_hft_data.md); references are in §11.

---

## 0. Goal and anchor paper

**Claim to test:** a large order-book model is too slow to act on, and a small model distilled from it keeps
most of its predictive value at a latency it can actually meet.

**Anchor paper (course requirement):** DeepLOB, Zhang, Zohren & Roberts (2019), [arXiv 1808.03668](https://arxiv.org/abs/1808.03668).
- **Result to reproduce:** DeepLOB's accuracy / precision / recall / F1 on the FI-2010 benchmark, Setup 2
  (train days 1-7, test days 8-10), horizons k = 10, 20, 50: paper Table II, F1 83.40 / 72.82 / 80.35 %.
  (k = 10, 50, 100 are the Setup 1 horizons, Table I.) Paper model: conv 16 filters, Inception@32, LSTM 64,
  ~60k params; Adam lr 0.01, epsilon 1, batch 32, stop after 20 epochs without validation-accuracy gain.
  Paper uses the z-score FI-2010 files; ours are the decimal-precision files from the authors' repo. FI-2010 is free and small, so this is about a week of work and is
  the project's fallback if the market-making extension slips.
- **Our extension:** (1) re-run the DeepLOB setup on our own Nasdaq ITCH data, pooled across stocks;
  (2) distil it into a cache-sized student; (3) measure each model's inference latency; (4) show how much of
  the predictive edge, and of simulated market-making PnL, survives once latency is charged.

| Gate | Hypothesis | Test |
|---|---|---|
| 1 | Large models (DeepLOB, big MLP) beat linear and LightGBM baselines on held-out prediction when latency is ignored. | Accuracy, macro-F1, IC on the test day |
| 2 | At realistic latency, the distilled student beats (a) the teacher charged its own latency, (b) a same-size student trained directly, (c) LightGBM. | Prediction and PnL vs latency curves; crossover latency |

Gate 1 may fail: the literature is mixed on deep vs simple models at short horizons ([Briola 2020](https://arxiv.org/abs/2007.07319),
[TLOB 2025](https://arxiv.org/abs/2502.15757), [Briola 2024](https://arxiv.org/abs/2403.09267)). If it does, that is reported as a result.

---

## 1. Data and split

- **Source:** free Nasdaq TotalView-ITCH 5.0 sample files from emi.nasdaq.com (full L3: every add, cancel,
  replace and execution with order IDs). No Databento purchase needed.
- **Days:** Dec 8–12 2025 (five consecutive trading days).
  - train Dec 8–10, validate Dec 11 (early stopping only), test Dec 12. Chronological, no shuffling.
    Labels never cross a day boundary, so the overnight gap separates the splits.
  - **Extra out-of-time test:** the free May 15, May 18 and Jun 12 2026 files, if time allows.
- **Symbols:** QQQ plus two mid-liquidity Nasdaq-listed stocks, chosen by `code/data/select_symbols.py`
  from one day of data: at least 20k trades/day, mean spread ≤ 3 ticks, and the shortest best-level queue
  measured in trades. Short queues mean our own simulated orders actually get filled.
- **One pooled model, not one per stock.** All features are scale-free (ticks, ratios, values relative to the
  stock's own trailing 5-minute average) and the symbol id is an input. Pooling gives each model several
  times more data; [Sirignano & Cont 2019](https://arxiv.org/abs/1803.06917) find pooled models beat per-stock ones.
- **Storage:** raw files (~9 GB/day) and all derived data live on the external drive
  (`HFT_DATA_DIR=/Volumes/SeagateData/hft_data`), never in OneDrive or the repo.
- **Session:** 09:35–15:55 ET (first and last five minutes dropped; they also warm up trailing windows).

---

## 2. Pipeline (built 2026-11-03)

```
emi.nasdaq.com  ──download_itch.py──>  itch_raw/S120825-v50.txt.gz        (parallel ranged download)
                ──itch_parse.py────>  itch_events/<day>/<SYM>.parquet      (numba ITCH 5.0 parser, A/F/E/C/X/D/U/P)
                ──book.py──────────>  book/<day>/<SYM>.parquet             (L3 book -> 100 ms top-10 snapshots + flow)
                ──features.py──────>  features/<day>/<SYM>.parquet         (~40 features + labels)
                ──models/baselines.py──> results/baselines/
```

Book checks: no crossed snapshots and zero unmatched order references on the test sample. TODO: compare
our best bid/offer against an independent source for one day.

---

## 3. Prediction task

- **Update 2026-11-03: main target is now event-time.** Rows at every best bid/ask change, label = direction
  of the next mid change (see `results/event_time_results.md`: accuracy 0.66 vs ~0.53 directional on the
  1 s grid label). The 100 ms grid below is kept as the comparison setup.
- **Decision grid:** every 100 ms (~228k rows per stock per day).
- **Label:** mid-price change over h ∈ {1 s, 5 s, 10 s} (h = 10, 50, 100 steps): down / flat / up, plus the
  raw change in ticks for regression. **No smoothed labels**: DeepLOB/FI-2010-style smoothed means leak future
  information ([Briola 2024](https://arxiv.org/abs/2403.09267) discusses this).
- **Features (baselines and MLPs):** spread; queue imbalance; depth imbalance at 3/5/10 levels; microprice
  deviation; level gaps; depth relative to trailing average; order-flow imbalance
  ([Cont et al. 2014](https://arxiv.org/abs/1011.6402)) over 0.1/1/5/30 s; mid returns over 0.1 s–5 min; recent
  absolute movement; trade-flow imbalance and relative volume over 1/5/30 s; activity; last trade vs mid;
  time of day; symbol id.
- **Raw input (DeepLOB):** the last 100 snapshots of the 10-level book (40 values each), normalised per stock
  with trailing statistics, as in the paper.
- **Metrics:** accuracy, macro-F1, directional accuracy when both prediction and outcome move, and
  information coefficient (correlation of predicted and realised move) for the regression variants.

---

## 4. Models

| Role | Model | Notes |
|---|---|---|
| Baselines | majority class; multinomial logistic regression / ridge; LightGBM (classifier + Huber regressor) | done first, `code/models/baselines.py` |
| Student | MLP, ~2–10k params, on the engineered features | sized to fit in L1 cache |
| Teacher 1 | large MLP on the same features | tens of µs |
| Teacher 2 | DeepLOB (CNN + inception + LSTM, ~140k params) on raw book history | 100s of µs; anchor-paper model |
| Teacher 3 (optional) | small Transformer over recent snapshots (TLOB-style) | only if time allows |

Training runs locally on the M1 Max (Apple GPU via PyTorch MPS). Estimated: student minutes, big MLP
~30 min, DeepLOB ~2–4 h (overnight).

---

## 5. Distillation

| # | Method | What the student learns |
|---|---|---|
| D1 | Soft-label KD ([Hinton 2015](https://arxiv.org/abs/1503.02531)) | teacher class probabilities at temperature τ, mixed with hard labels |
| D2 | Regression KD | teacher's predicted mid change (smoothed estimate of E[Δmid \| s]) |
| D5 | Teacher-labelled extra data ([Born-Again Nets](https://arxiv.org/abs/1805.04770)) | teacher labels on extra symbols / days with no extra cost |
| D6 | **Latency-matched teacher (ours)** | teacher trained to predict the move from t+L to t+L+h using features at t, i.e. the target the student faces when it acts L later |

**Teacher and student get the same information (decided 2026-11-03).** The teacher's edge must come from
capacity, not from seeing more: both see the last N book updates; the teacher processes the full raw window,
the student a few summary features of the same window that it can compute in ~1 µs. Cheap extra inputs
(e.g. cross-asset lead–lag) go to both. D4 (privileged information) is only a labelled side ablation, since
we have no data that is genuinely unusable in time. If the teacher does not beat the student, that is the
result (equal accuracy at far lower latency). Details: `results/event_time_results.md`.

**Why distillation can help:** a single realised mid move is mostly noise; the teacher's output is a smoother
target. Ablations: student direct vs D1 vs D2 vs D2+D5 vs D2+D5+D6, against LightGBM at matched latency.
Dropped: hint/feature matching (D3) and privileged inputs (D4), unless time allows.

---

## 6. Trading evaluation (hftbacktest)

- **Simulator:** [hftbacktest](https://github.com/nkaz001/hftbacktest) (free, MIT). It supports separate feed and
  order latency and queue-position models, and replays L3 data. We write a converter from our ITCH event
  parquet to its event format. Non-reactive replay with 1-lot orders: our orders don't move the market (stated as an assumption).
- **Strategy:** a simple quoting rule shared by all models: quote both sides at the touch, skew or pull the side
  the model predicts will be run over, and cap inventory. Only the predictor changes between runs.
- **Latency:** total latency = feed + order latency (swept) + each model's **measured** inference time.
- **Queue-position scenarios:** results are reported under assumed queue positions (back of queue,
  hftbacktest's probabilistic power-law model, front 10%). A front-of-queue spot is what being fast buys at
  a new price level, so this links back to latency. **Phrase it as a scenario:** "PnL under assumed queue
  position X, which needs latency Y", never as achieved profit.
- **Fees:** rebate / zero / taker-fee scenarios from the Nasdaq price list.

---

## 7. Latency measurement

- Batch-1, single-thread CPU inference: p50/p99/p99.9 per model. Teachers in PyTorch and ONNX Runtime; the
  student as hand-written C (or numba) with int8/fp16 weights.
- Development numbers on the M1 Max are labelled as such. A Linux x86 run is a nice-to-have, not required.

---

## 8. Figures for the poster

1. DeepLOB on FI-2010: our reproduction vs the paper.
2. Prediction quality (macro-F1 / IC) on ITCH: baselines, teachers, student direct, student distilled.
3. **Headline:** PnL per day vs total latency, one curve per model, each model's own latency marked.
4. Queue-position scenarios at a fixed latency.
5. Latency table (p50/p99) and model size vs L1/L2 cache.

---

## 9. Schedule (presentation Dec 8/10, report Dec 13, repo Dec 15)

| Week | Work |
|---|---|
| Nov 3–9 | ITCH pipeline, symbol selection, linear/LightGBM baselines; start DeepLOB on FI-2010 |
| Nov 10–16 | Finish FI-2010 reproduction; teachers (big MLP, DeepLOB on ITCH); direct student |
| Nov 17–23 | Distillation D1/D2/D5/D6; latency harness |
| Nov 24–30 | hftbacktest converter and quoting rule; latency sweep; queue scenarios; out-of-time days |
| Dec 1–7 | Figures, poster, rehearsal |
| Dec 8–15 | Present; 2-page report; README and repo cleanup |

Commit small and often: the repo grade is based on commit history.

---

## 10. Cut from the previous plan

- Own replay engine and counterfactual all-action reward labels (replaced by hftbacktest + a fixed quoting rule).
- RL (PPO) and DAgger.
- Databento purchase (free ITCH files cover the need). `code/archive/opra_slice.py` kept for reference only.
- LLM trading arm: at most one slide as an infeasibility note (0.5–3 s per decision vs a µs loop).

---

## 11. References

**Market making and microstructure**
- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*: https://doi.org/10.1080/14697680701381228 (DOI resolves)
- Guéant, Lehalle & Fernandez-Tapia (2013), *Dealing with the inventory risk*: https://arxiv.org/abs/1105.3115 **[V]**
- Moallemi & Yuan, *A model for queue position valuation in a limit order book* (Columbia working paper): https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/25461/queue-value-2016.pdf **[S]**
- Stoikov (2018), *The micro-price*, Quantitative Finance 18(12): https://ideas.repec.org/a/taf/quantf/v18y2018i12p1959-1966.html **[S]**
- Cont, Kukanov & Stoikov (2014), *The price impact of order book events* (OFI): https://arxiv.org/abs/1011.6402 **[V]**
- Kolm, Turiel & Westray (2023), *Deep order flow imbalance*: https://papers.ssrn.com/abstract=3900141

**DL on limit order books**
- Zhang, Zohren & Roberts (2019), *DeepLOB*: https://arxiv.org/abs/1808.03668 **[V]**
- Wallbridge (2020), *TransLOB*: https://arxiv.org/abs/2003.00130
- Berti & Kasneci (2025), *TLOB*: https://arxiv.org/abs/2502.15757
- Briola, Turiel & Aste (2020), *DL modelling of LOB: a comparative perspective*: https://arxiv.org/abs/2007.07319
- Briola, Bartolucci & Aste (2024), *Deep LOB forecasting (LOBFrame)*: https://arxiv.org/abs/2403.09267
- Sirignano & Cont (2019), *Universal features of price formation*: https://arxiv.org/abs/1803.06917
- Ke et al. (2017), *LightGBM*: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

**RL and bandits for market making**
- Spooner et al. (2018), *Market making via RL*: https://arxiv.org/abs/1804.04216 **[V]**
- Spooner & Savani (2020), *Robust market making via adversarial RL*: https://arxiv.org/abs/2003.01820
- Relaver (2025), *Latency and inventory risk in MM with RL*: https://arxiv.org/abs/2505.12465
- Dudík, Langford & Li (2011), *Doubly robust policy evaluation and learning*: https://arxiv.org/abs/1103.4601 **[V]**
- Schulman et al. (2017), *PPO*: https://arxiv.org/abs/1707.06347 **[V]**

**Distillation**
- Hinton, Vinyals & Dean (2015), *Distilling the knowledge in a neural network*: https://arxiv.org/abs/1503.02531 **[V]**
- Rusu et al. (2015), *Policy distillation*: https://arxiv.org/abs/1511.06295 **[V]**
- Romero et al. (2014), *FitNets*: https://arxiv.org/abs/1412.6550 **[V]**
- Lopez-Paz et al. (2015), *Unifying distillation and privileged information*: https://arxiv.org/abs/1511.03643 **[V]**
- Furlanello et al. (2018), *Born-again neural networks*: https://arxiv.org/abs/1805.04770 **[V]**
- Ross, Gordon & Bagnell (2011), *DAgger*: https://arxiv.org/abs/1011.0686 **[V]**

**Closest prior work (latency and distillation in trading)**
- Fu et al. (2026), *Distilling LLM features into small models for market making*: https://arxiv.org/abs/2511.07110
- Hedges (2026), *FastBiNLOB*: https://arxiv.org/abs/2606.25986

**Tools**
- hftbacktest L3 backtesting: https://hftbacktest.readthedocs.io/en/latest/tutorials/Level-3%20Backtesting.html
