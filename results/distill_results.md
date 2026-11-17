# Distillation into the 4.6k-parameter MLP

*2026-11-16. Code: `code/models/distill.py`. Raw numbers: `distill/metrics.json`.*

## Setup

- **Student:** the small MLP (34 hand-made features + symbol embedding -> 64 -> 32 -> 2; 4,654 params;
  10 us batch-1 CPU inference). Same rows, label (direction of the next mid change) and split as before.
- **Target:** alpha * true label + (1 - alpha) * teacher P(up), soft cross-entropy. alpha = 1 is the direct
  baseline; alpha = 0.5 mixes; alpha = 0 is teacher only.
- **Selection:** early stopping on validation loss against the true labels for every variant; test day once.
  3 seeds per variant (mean, [min, max]).
- **Teachers:**
  - DeepLOB (raw 100-update book window): train-row outputs are in-sample (train acc 0.677 vs val 0.669).
  - LightGBM (same 34 features): cross-fitted on the training days, so its soft labels are out-of-sample.
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
   DeepLOB's extra edge comes from patterns in the raw book that are not functions of the student's 34
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

---

# Follow-up: students that also see a raw-book window

*2026-11-17. Code: `code/models/students.py`, `code/models/distill_raw.py`. Raw numbers: `distill_raw/metrics.json`.*

Point 3 above, tested. Two students get the 34 features **plus** the last 10 book updates of the top 5 levels
(200 values; prices as ticks from the current mid, sizes z-scored as for DeepLOB):

- **Raw MLP:** features + flattened window -> 64 -> 32 -> 2 (17,454 params; 13.7 us batch-1 CPU, TorchScript).
- **Tiny conv:** two Conv1d layers (20 -> 8 -> 8 channels, kernel 3) over the 10 updates, flattened and
  concatenated with the features -> 64 -> 32 -> 2 (8,414 params; 21.9 us).

Same teachers, alpha 0.5, selection rule and 3 seeds as above.

| Student | Teacher | Accuracy (3 seeds) | AUC | Log-loss | Val loss |
|---|---|---|---|---|---|
| Raw MLP | – | 0.6606 [0.6594, 0.6613] | 0.7254 | 0.6103 | 0.6011 |
| Raw MLP | DeepLOB | 0.6616 [0.6599, 0.6627] | 0.7260 | 0.6097 | 0.6003 |
| Raw MLP | LightGBM | 0.6624 [0.6622, 0.6625] | 0.7274 | 0.6079 | 0.5996 |
| Raw MLP | Average | 0.6631 [0.6624, 0.6635] | 0.7285 | 0.6074 | 0.5987 |
| Tiny conv | – | 0.6611 [0.6603, 0.6620] | 0.7264 | 0.6089 | 0.6002 |
| Tiny conv | DeepLOB | 0.6635 [0.6618, 0.6653] | 0.7284 | 0.6078 | 0.5984 |
| Tiny conv | LightGBM | 0.6637 [0.6628, 0.6647] | 0.7292 | 0.6064 | 0.5983 |
| **Tiny conv** | **Average** | **0.6652 [0.6646, 0.6659]** | **0.7309** | **0.6054** | **0.5967** |
| *LightGBM / DeepLOB themselves* | | *0.6629 / 0.6632* | *0.7277 / 0.7260* | *0.6077 / 0.6114* | |

**Reading**

1. **The raw window carries information on its own:** +0.4 points without any teacher (0.661 vs 0.657).
2. **With raw inputs, the most accurate teacher becomes the most useful one.** The DeepLOB + LightGBM average
   was the *worst* teacher for the features-only student and is now the *best* for both raw students, and the
   DeepLOB teacher now helps (+0.24 points for the tiny conv). This confirms the explanation in point 2: a
   student can only absorb what its inputs can express.
3. **A distilled 8.4k-parameter student beats both big models on accuracy:** tiny conv 0.6652 vs LightGBM
   0.6629 and DeepLOB 0.6632, at 22 us instead of 27 us and 1,764 us. (Variant chosen by validation loss.)

**With latency charged** (ticks saved per 1,000 decisions):

| Own infrastructure latency | 10 us | 100 us | 1 ms |
|---|---|---|---|
| Small MLP, distilled (10.5 us) | **462** | 385 | 247 |
| Raw MLP, distilled (13.7 us) | 447 | 384 | 246 |
| Tiny conv, distilled (21.9 us) | 432 | 384 | 247 |
| LightGBM (27 us) | 424 | 380 | 247 |
| DeepLOB, CPU (1,764 us) | 228 | 226 | 215 |

4. **At co-located speeds the extra microseconds cost more than the extra accuracy buys:** the tiny conv's
   +0.34 points over the distilled small MLP are outweighed by its 11 us of extra inference (432 vs 462 at
   10 us own latency). From ~100 us of own latency the students are equivalent (384-385).
5. **Caveat: these latencies are mostly PyTorch overhead.** All three students need ~0.01-0.03 MFLOPs, under
   1 us of arithmetic on one core. Hand-written C (plan item, section 7) would remove most of the gap, and
   then the most accurate student (tiny conv) would win at every latency. Measuring that is the next step.
