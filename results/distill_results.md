# Distillation into the 4.6k-parameter MLP

*2026-11-16. Code: `code/models/distill.py`. Raw numbers: `distill/metrics.json`.*

## Setup

- **Student:** the small MLP (33 hand-made features + symbol embedding -> 64 -> 32 -> 2; 4,610 params;
  10 us batch-1 CPU inference). Same rows, label (direction of the next mid change) and split as before.
- **Target:** alpha * true label + (1 - alpha) * teacher P(up), soft cross-entropy. alpha = 1 is the direct
  baseline; alpha = 0.5 mixes; alpha = 0 is teacher only.
- **Selection:** early stopping on validation loss against the true labels for every variant; test day once.
  3 seeds per variant (mean, [min, max]).
- **Teachers:**
  - DeepLOB (raw 100-update book window): train-row outputs are in-sample (train acc 0.677 vs val 0.669).
  - LightGBM (same 33 features): cross-fitted on the training days, so its soft labels are out-of-sample.
  - Average of the two (most accurate teacher: val 0.677).

## Results (test day Dec 12)

| Student variant | Teacher | alpha | Accuracy (3 seeds) | AUC | Log-loss |
|---|---|---|---|---|---|
| Direct | – | 1 | 0.6569 [0.6566, 0.6575] | 0.7208 | 0.6130 |
| Distilled | DeepLOB | 0.5 | 0.6584 [0.6573, 0.6593] | 0.7219 | 0.6127 |
| Distilled | DeepLOB | 0 | 0.6540 [0.6537, 0.6543] | 0.7156 | 0.6181 |
| **Distilled** | **LightGBM** | **0.5** | **0.6618 [0.6615, 0.6620]** | **0.7262** | **0.6089** |
| Distilled | LightGBM | 0 | 0.6615 [0.6613, 0.6617] | 0.7256 | 0.6095 |
| Distilled | Average | 0.5 | 0.6607 [0.6602, 0.6610] | 0.7249 | 0.6101 |
| Distilled | Average | 0 | 0.6602 [0.6602, 0.6604] | 0.7240 | 0.6113 |
| *Reference: LightGBM itself* | | | *0.6629* | *0.7277* | *0.6077* |
| *Reference: DeepLOB itself* | | | *0.6632* | *0.7260* | *0.6114* |

## Reading

1. **Distillation works: +0.49 points for free at inference.** With LightGBM soft labels the 4.6k student goes
   from 0.6569 to 0.6618, closing ~80% of its gap to LightGBM (0.6629) at the same 10 us inference. Seed
   ranges do not overlap.
2. **The best teacher is the one whose knowledge the student can express, not the most accurate one.** The
   DeepLOB + LightGBM average is the most accurate teacher (val 0.677) but gives a worse student (0.6607)
   than LightGBM alone (0.6618); DeepLOB alone helps least (0.6584) and, as the only target, hurts (0.6540).
   DeepLOB's extra edge comes from patterns in the raw book that are not functions of the student's 33
   features, so the student cannot copy them and they act as noise. LightGBM is a function of exactly those
   features, so its soft labels are learnable. (DeepLOB's in-sample train outputs may add over-confidence.)
3. **Consequence for the latency story.** To pass DeepLOB's raw-book edge on, the student needs inputs that
   carry it: e.g. a few extra raw-window features or a tiny 1-D conv over the last updates, at a small latency
   cost. That is the next distillation experiment.

## With latency charged (simulator, ticks saved per 1,000 decisions)

| Own infrastructure latency | 10 us | 100 us | 1 ms |
|---|---|---|---|
| Small MLP, direct (10 us) | 455 | 379 | 243 |
| **Small MLP, distilled from LightGBM (10 us)** | **463** | **385** | **247** |
| LightGBM (27 us) | 424 | 380 | 247 |
| DeepLOB, CPU (1.76 ms) | 228 | 226 | 215 |

The distilled student is the best CPU model at every infrastructure latency up to ~1 ms, where all models
converge.
