# Implementation Plan: Distilled Low-Latency Market Making on QQQ

Rewritten 2026-10-11 for the QQQ equity market-making design. Data is in [free_hft_data.md](free_hft_data.md). Paper links are in §10. **[V]** = link/title checked 2026-10-11; **[S]** = secondary source; untagged design choices are ours.

---

## 0. Goal

Show that in market making, a **large model is too slow to use** and a **small distilled model recovers most of its value at a latency it can actually meet**.

| Gate | Hypothesis | Test |
|---|---|---|
| 1 | A large DL policy beats simpler policies (Avellaneda–Stoikov, linear, LightGBM, small MLP) on out-of-sample simulated PnL **when latency is ignored**. | PnL at zero latency, held-out days |
| 2 | At realistic latency, a **distilled** small model beats (a) the large teacher run at its own latency, (b) the same-size model trained directly, and (c) LightGBM. | PnL vs latency curves; the crossover latency |

If gate 1 fails, report it. The literature is mixed on whether DL beats simple models at short horizons ([Briola 2020](https://arxiv.org/abs/2007.07319), [TLOB 2025](https://arxiv.org/abs/2502.15757), [Briola 2024](https://arxiv.org/abs/2403.09267)).

---

## 1. Data and split

- **Data:** QQQ MBO from Databento `XNAS.ITCH` ($1.20/GB, older history available). Add the large constituents (NVDA, AAPL, MSFT) for cross-asset features if affordable.
- **Amount:** aim for **2–3 months of consecutive trading days**.
- **Split (chronological, no shuffling):**
  - train on the first ~60% of days;
  - validate on the next ~20%;
  - test on the last ~20%.
  - Leave a gap of at least one day between splits.
- **Extra out-of-time test:** the free Nasdaq ITCH days (Dec 2025, May/Jun 2026) if they fall outside the bought range.
- **Session:** regular hours only. Drop the first and last 5 minutes (auctions).

---

## 2. Simulator: event replay with our own orders

```
Databento MBO (.dbn.zst) ──> L3 book: price level -> FIFO queue of (order_id, size)
                                   │
   for each exchange event e (timestamp order):
     1. activate our orders/cancels whose arrival time <= e.ts
     2. apply e; executions/cancels of orders AHEAD of ours shrink shares_ahead;
        an execution that reaches our order -> FILL
     3. at decision times t: model sees book(t − feed_lat), decides;
        action reaches exchange at t + compute_lat(model) + order_lat
                                   │
   per fill: price, side, markouts at +1ms/+10ms/+100ms/+1s, inventory, fees/rebates
```

- **Queue position:**
  - Our new order joins the **back** of its price level when it arrives.
  - MBO order IDs tell us exactly which executions and cancels were ahead of us.
- **Picked off:** if the price trades through our level before our cancel arrives, we are filled. This is how latency costs money.
- **Non-reactive replay:** our 1-lot orders don't change other traders' behaviour. This is standard ([hftbacktest](https://hftbacktest.readthedocs.io/en/latest/tutorials/Level-3%20Backtesting.html) **[S]**); state it in the write-up.
- **Latency:**
  - `compute_lat` is each model's **measured** inference time (C++ harness for students, PyTorch/ONNX for teachers).
  - Total latency is swept from 1 µs to 10 ms.
- **Fees:** Nasdaq maker rebate and taker fee, looked up from the current price list (TODO). Run rebate / zero / fee scenarios.
- **Build:** our own replay engine (numba or C++), so it can produce the counterfactual labels in §3. Validate it in two ways:
  - our rebuilt best bid/offer must match Databento's MBP-1 at every event;
  - the final backtests are re-run in hftbacktest, which reads Databento MBO directly.

---

## 3. Problem formulation

**Fair value = mid (or microprice).** Prediction isn't modelled separately. The model learns it implicitly through the markout term, i.e. which fills turn out toxic.

**State s:**
- **Book:** queue imbalance, [microprice](https://ideas.repec.org/a/taf/quantf/v18y2018i12p1959-1966.html), and [order-flow imbalance](https://arxiv.org/abs/1011.6402) over several windows.
- **Activity:** trade-sign flow, short-horizon momentum and volatility.
- **Cross-asset:** returns and order-flow imbalance of the constituents.
- **Our position:**
  - our orders' queue position and the shares ahead of them;
  - inventory q;
  - time of day.
- **Feed-latency rule:** every feature is computed as of t − feed latency.

**Actions, chosen per side and independently:** keep resting order / cancel / join at best (back of queue) / quote one tick behind / don't quote. On QQQ the spread is almost always 1 tick and queues are huge. The key decision is therefore **whether to keep our place in the queue**, which is valuable and can't be rebuilt once cancelled ([Moallemi & Yuan](https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/25461/queue-value-2016.pdf) **[S]**). Actions that would push |q| past Q_max are masked out.

**Reward of action a at time t (counterfactual, from replay):**
```
V_a(t) = Σ_fills [ side · (mid_{t+h} − fill_price) + rebate − fee ]     # order arrives at t + L
```

**Why this is mostly not RL.**
- **Every action can be scored.** Our orders don't affect the replay, so V_a can be computed for **every** action at every decision point, not just the action taken. That is a full-information contextual bandit: learning is close to supervised and avoids RL's variance ([Dudík et al. 2011](https://arxiv.org/abs/1103.4601) **[V]** covers the general off-policy setting).
- **Inventory is the only link between decisions.** Its long-run cost is approximately quadratic, γσ²q², which is the standard market-making result ([Avellaneda & Stoikov 2008](https://doi.org/10.1080/14697680701381228); [Guéant et al. 2013](https://arxiv.org/abs/1105.3115) **[V]**). So we:
  - put q into the state;
  - randomise q during training so the model sees all inventory levels;
  - add the quadratic penalty to the loss.
- **Full RL is the stretch goal (§7).**

**Objective:** maximise E[V], not E[V | fill]. Maximising E[V | fill] alone would teach the model to quote wide and rarely trade.

---

## 4. Loss: direct E[V] maximisation

```
π_θ(a|s) = softmax(g_θ(s)/τ)                       # one head per side
loss(θ) = − Σ_a π_θ(a|s) · V_a(t)                  # expected PnL under the policy
          + γ σ² · E_π[(q + Δq_a)²]                # inventory risk
          − η · H(π_θ)                              # entropy; anneal to 0
```

- **Value head:** add a head Q_θ(s, a) regressed on V_a with a Huber loss. It stabilises training, and distillation (§6) needs it.
- **Inference:** take the argmax action, or argmax of Q.

---

## 5. Models

| Role | Model | Size / latency target |
|---|---|---|
| Rule baseline | Avellaneda–Stoikov with microprice | ns |
| Baselines | linear; [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) (one regressor per action predicting V_a) | µs |
| Student | MLP, ~2–10k params, int8/fp16 | fits in L1 cache, sub-µs to ~1 µs in C++ |
| Teacher 1 | large MLP | tens of µs |
| Teacher 2 | CNN-LSTM over the book ([DeepLOB](https://arxiv.org/abs/1808.03668) **[V]**) | 100s of µs |
| Teacher 3 | Transformer over the last N events ([TLOB](https://arxiv.org/abs/2502.15757), [TransLOB](https://arxiv.org/abs/2003.00130)) | 100s of µs to ms |

All models use the same inputs, actions and loss; only capacity and latency differ. Teachers may also use inputs that are too slow to compute live, such as a long event history or extra cross-asset inputs.

---

## 6. Distillation methods

| # | Method | What the student learns | Source |
|---|---|---|---|
| D1 | Policy KD | Teacher's action distribution, softened by temperature τ | [Hinton 2015](https://arxiv.org/abs/1503.02531), [Policy Distillation](https://arxiv.org/abs/1511.06295) **[V]** |
| D2 | **Q-value distillation** | Teacher's Q_T(s, a) for **every** action | Policy Distillation (Q-regression variant) **[V]** |
| D3 | Hint / feature matching | Teacher's hidden features | [FitNets](https://arxiv.org/abs/1412.6550) **[V]** |
| D4 | Privileged teacher | Teacher sees inputs the student can't compute live | [Lopez-Paz 2015](https://arxiv.org/abs/1511.03643) **[V]** |
| D5 | Teacher-labelled extra data | Teacher labels unlabelled states: more days, randomised inventory, other symbols | [Born-Again Nets](https://arxiv.org/abs/1805.04770) **[V]** (same-size students beat their teachers) |
| D6 | **Latency-matched teacher (ours)** | Teacher trained on V_a computed at the **student's** latency, so it learns the policy that is optimal at the speed the student will run | none found |

**Combined student loss:**
```
L_S = α · L_bandit(V_a)                                       # §4, raw counterfactual labels
    + (1 − α) · [ Huber(Q_S(s,·), Q_T(s,·)) + τ² · KL(π_T^τ ‖ π_S^τ) ]
    + λ_h · ‖W f_S − f_T‖²                                     # optional D3
```

**Why distillation should beat direct training (the gate 2 argument).**
- Raw V_a is mostly noise: a single markout is dominated by random price moves.
- The teacher's Q_T(s, a) is a smoothed estimate of E[V_a | s], so the student fits cleaner targets.
- D5 then provides those clean targets on far more states than the labelled data covers.

**Ablations:** D1 alone, D2 alone, D2 + D5, D2 + D5 + D6. Each is compared against the same-size direct student and LightGBM.

**Prior work closest to this:**
- [Fu et al. 2026](https://arxiv.org/abs/2511.07110) distil an LLM into small market-making models.
- [Hedges 2026](https://arxiv.org/abs/2606.25986) studies latency-efficient LOB models.
- [Relaver 2025](https://arxiv.org/abs/2505.12465) does RL market making under latency.

None of them distils a latency-matched teacher evaluated in a queue-exact simulator.

---

## 7. RL (stretch)

- **Method:** PPO ([Schulman 2017](https://arxiv.org/abs/1707.06347) **[V]**) on the simulator over whole-day episodes, initialised from the bandit policy. Market-making RL references: [Spooner 2018](https://arxiv.org/abs/1804.04216) **[V]** and [Spooner & Savani 2020](https://arxiv.org/abs/2003.01820).
- **Question it answers:** does full RL beat bandit training with the quadratic inventory penalty? If not, the bandit approach is justified.
- **Distilling an RL policy:**
  - start with D1/D2 on teacher rollouts;
  - then [DAgger](https://arxiv.org/abs/1011.0686) **[V]**: run the student, have the teacher label the states the student actually visits, retrain. This fixes the mismatch where the student drifts into inventory states the teacher rarely visited.

---

## 8. Evaluation

- **Headline plot:** PnL per day vs total latency, one curve per model. Mark each model's own measured latency on its curve. The result is the point where the distilled student beats the teacher run at the teacher's own latency.
- **Per-fill markouts:** at +1 ms / +10 ms / +100 ms / +1 s, split into "picked off before our cancel arrived" and normal fills.
- **Other metrics:** fill rate, inventory variance, Sharpe across test days, and bootstrap confidence intervals over days.
- **Latency table:** p50/p99 inference time per model (C++ on Linux x86 for final numbers; Mac numbers for development only).

---

## 9. Implementation order

1. Pull 1 day of QQQ MBO. Build the book and validate it against MBP-1.
2. Write the simulator with our orders, queue tracking and latency. Run the Avellaneda–Stoikov baseline.
3. Write the counterfactual label generator: V_a for every action at the sampled decision points.
4. Train the baselines (linear, LightGBM) and the direct MLP student on §4.
5. Train the teachers (MLP → CNN-LSTM → Transformer). Gate 1 check.
6. Run distillation D1–D6, plus the C++ inference harness for latency.
7. Run the latency sweep and plots. Gate 2 check.
8. Stretch: PPO and DAgger.

---

## 10. References

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
