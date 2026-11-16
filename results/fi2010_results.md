# DeepLOB Reproduction on FI-2010 (Setup 2)

*2026-11-16. Code: `code/models/deeplob.py`, `code/models/train_fi2010.py`. Raw numbers and per-epoch
history: `fi2010/deeplob_k{10,20,50}.json`.*

## Setup

- **Target:** Zhang, Zohren & Roberts (2019), Table II (FI-2010 Setup 2): train days 1-7, test days 8-10.
- **Data:** FI-2010 (no auction), decimal-precision normalised files from the authors' repository.
  The paper used the z-score files and reports only a "subtle difference" between normalisations.
- **Model:** as published: conv blocks with 16 filters, Inception@32, LSTM(64); 61.6k parameters.
- **Training:** as published: Adam (lr 0.01, epsilon 1), batch 32, early stopping after 20 epochs without
  validation-accuracy gain. The paper does not say where the FI-2010 validation set comes from; as in the
  authors' code, the last 20% of the training days is held out. Test days are used once, at the end.
- **Metrics:** precision / recall / F1 weighted by class frequency. The paper's Table II has recall equal to
  accuracy in every row, which only holds for the weighted average. Macro averages are also reported.
- **Compute:** Apple M1 Max GPU (MPS), three horizons in parallel, ~2 h each.

## Results (blind test, days 8-10)

| k | Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|---|---|---|---|---|---|---|
| 10 | Paper | 84.47 | 84.00 | 84.47 | 83.40 | – |
| 10 | Ours | 83.06 | 82.17 | 83.06 | 82.10 | 72.13 |
| 20 | Paper | 74.85 | 74.06 | 74.85 | 72.82 | – |
| 20 | Ours | 73.73 | 72.38 | 73.73 | 72.11 | 63.61 |
| 50 | Paper | 80.51 | 80.38 | 80.51 | 80.35 | – |
| 50 | Ours | 78.38 | 78.33 | 78.38 | 78.32 | 76.74 |

Best epoch 34 / 36 / 26 (k = 10 / 20 / 50); stopped at 55 / 57 / 47. Test class balance
(down / stationary / up): k=10 0.15 / 0.71 / 0.14; k=20 0.20 / 0.62 / 0.18; k=50 0.28 / 0.47 / 0.25.

## Reading

- **Reproduced within 0.7-2.1 F1 points** at every horizon, including the paper's non-monotone pattern
  (k = 20 is hardest, k = 50 easier than k = 20).
- **Weighted vs macro F1 matters.** At k = 10, 71% of test labels are "stationary"; weighted F1 is 82.1
  but macro F1 is 72.1. Comparisons with DeepLOB's published numbers must use the weighted average.
- **Remaining gap, likely causes:** decimal-precision instead of z-score normalisation; single seed;
  validation set taken from the end of training (shorter training set than the paper may have used);
  PyTorch vs Keras defaults (initialisation, BatchNorm momentum).
- **Caveat (from the paper's own setup, kept for fidelity):** FI-2010 labels average the next k mid-prices,
  so they use future information and are not directly tradable; our Nasdaq extension uses unsmoothed labels.

## Next

- Seeds (3-5) for error bars; optionally the z-score files.
- DeepLOB on the event-time Nasdaq ITCH data (the capacity teacher) vs LightGBM and the 4.5k MLP.
