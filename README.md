# Fast Enough to Matter: Distilling Order-Book Models for Low-Latency Market Making

STAT UN3108: Applied Deep Learning and AI, Columbia University, Fall 2026

**Group members:** Jack Fanshawe (add others)

## 1. Introduction

This repository re-implements DeepLOB and investigates whether its predictive edge survives the latency constraints of market making. Large order-book models are too slow to act on at the speed quotes must be updated; we distil them into a small, cache-sized student and measure how much predictive value and simulated PnL each model keeps once its own inference latency is charged.

- **Selected paper:** Z. Zhang, S. Zohren, S. Roberts, *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*, IEEE Transactions on Signal Processing, 2019 ([arXiv 1808.03668](https://arxiv.org/abs/1808.03668)).
- **Main contribution of the paper:** a CNN + inception + LSTM network that learns features directly from raw limit-order-book snapshots and predicts mid-price direction, beating hand-crafted-feature models on the FI-2010 benchmark and transferring to unseen LSE stocks.

## 2. Live Demo (optional)

TODO: link to deployed demo and setup steps, if any.

## 3. Chosen Result

- **Result targeted for reproduction:** DeepLOB accuracy and macro-F1 on FI-2010 for prediction horizons k = 10, 50, 100.
- **Why it matters:** it is the paper's headline evidence that a deep model on raw book data beats feature-based models; our extension asks whether that advantage still holds on recent Nasdaq data, against strong tree baselines, and after latency is charged.
- **Reference in original paper:** FI-2010 results table (TODO: confirm table number from the paper).

## 4. GitHub Contents

| Path | Contents |
|------|----------|
| `README.md` | This file |
| `code/` | Re-implementation code, configs, and scripts |
| `data/` | Datasets, or a README on how to obtain them |
| `results/` | Generated figures, tables, and logs |
| `poster/` | PDF of the poster presented in class |
| `report/` | PDF of the final two-page report |
| `LICENSE` | MIT License |
| `.gitignore` | Files ignored by Git |
| `code/data/` | ITCH download, parser, order-book rebuild, features (`download_itch.py`, `itch_parse.py`, `book.py`, `features.py`, `select_symbols.py`) |
| `code/models/` | Baselines (`baselines.py`), MLPs (`mlp.py`), learning curve on both targets (`learning_curve.py`); DeepLOB and distillation to follow |
| `code/analysis/` | Figures (`plot_baselines.py`, `plot_learning_curve.py`) |
| `research/` | Literature notes and the implementation plan |
| `Propsoal.md` | Initial project proposal (superseded by `research/implementation_plan.md`) |
| `ProgressTracker.md` | Dated progress log |

## 5. Re-implementation Details

- **Approach:** reproduce DeepLOB on FI-2010; rebuild the full Nasdaq order book from raw ITCH messages, sample it every 100 ms, and predict mid-price direction 1 s / 5 s / 10 s ahead with one model pooled across stocks; distil large teachers into a small student; evaluate in hftbacktest with latency and queue-position scenarios.
- **Models:** logistic regression and LightGBM baselines; large MLP and DeepLOB teachers; ~2-10k-parameter MLP student (direct vs distilled).
- **Datasets:** FI-2010; free Nasdaq TotalView-ITCH 5.0 files for Dec 8-12 2025 (QQQ + two mid-liquidity stocks). See `data/README.md`.
- **Evaluation metrics:** accuracy, macro-F1, information coefficient; measured inference latency (p50/p99); simulated PnL vs latency.
- **Challenges / modifications to the original:** unsmoothed labels (smoothed FI-2010-style labels leak future prices); chronological day-level splits; latency charged per model.

## 6. Reproduction Steps

```bash
git clone <repo-url> && cd DeepLearningAIProjet
pip install -r requirements.txt
export HFT_DATA_DIR=/path/to/external/drive/hft_data      # ~50 GB free needed for the raw week
bash code/run_pipeline.sh                                  # download, parse, rebuild books, features
python code/models/baselines.py                            # linear + LightGBM baselines -> results/baselines/
```

- **Compute requirements:** any recent laptop. Developed on an Apple M1 Max (64 GB RAM). Parsing is ~5 min per day; baselines take minutes; DeepLOB training ~2-4 h on the Apple GPU.

## 7. Results / Insights

**Baselines (2026-11-03)**, pooled over QQQ, PYPL and NVDA, test day Dec 12 2025. Full write-up: [results/baseline_results.md](results/baseline_results.md).

| Horizon | Majority class | Logistic | LightGBM | MLP small (4.5k) | MLP large (413k) |
|---|---|---|---|---|---|
| 1 s | 0.18 | 0.53 | 0.54 | 0.53 | 0.53 |
| 5 s | 0.20 | 0.47 | 0.48 | 0.47 | 0.46 |
| 10 s | 0.21 | 0.43 | 0.42 | 0.42 | 0.40 |

*(macro-F1, down / flat / up)*

![Baseline comparison](results/baseline_comparison.png)

All learned models tie, and the large MLP is never better than the small one. Most of the gain is in predicting
*whether* the mid moves, not its direction (directional accuracy 0.53 at 1 s).

**Event-time target (2026-11-03).** Predicting the direction of the *next* mid change, sampled at every best
bid/ask update, lifts test accuracy to 0.66 (LightGBM 0.663, MLP 0.657, logistic 0.638, one-feature
queue-imbalance rule 0.626). Training on 1 → 3 days adds only 0.2–0.4 points. Model size still makes no
difference. The next mid change arrives a median 2–26 ms later for QQQ and NVDA, so latency matters directly.
Write-up: [results/event_time_results.md](results/event_time_results.md).

![Learning curve](results/learning_curve.png)

## 8. Conclusion

TODO: key takeaways and lessons learned.

## 9. References

1. TODO: original paper
2. TODO: tools, datasets, frameworks

## 10. Acknowledgements

This work was completed as part of the STAT UN3108 Applied Deep Learning and AI course at Columbia University (Fall 2026), taught by Kenta Takatsu and Chen Wang.
