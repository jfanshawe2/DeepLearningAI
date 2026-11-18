22/08: Create github repo, in idea generation phase currently

29/08 intial proposal put together
01/10: agent literature research (latency, distillation, DL for LOB, data, LLMs); repo setup

11/10: market-making design on QQQ; data plan (Databento QQQ MBO); implementation plan v1

03/11: scope cut for Dec 8 presentation (see research/implementation_plan.md):
- anchor paper set to DeepLOB (Zhang et al. 2019), reproduce FI-2010 results; distillation + latency is the extension
- data switched to free Nasdaq ITCH files (Dec 8-12 2025), QQQ + 2 mid-liquidity stocks, one pooled model
- hftbacktest for trading evaluation with queue-position scenarios; own replay engine, bandit labels and RL cut
- built pipeline: parallel downloader, numba ITCH parser, L3 book -> 100 ms snapshots, features/labels
- linear and LightGBM baselines
- data: all 5 days parsed (zero unmatched order refs, zero crossed books); symbols QQQ, PYPL, NVDA. Found that the
  Nasdaq-only book is unrepresentative for most mid/high-priced names, so selection requires a 1-tick spread >= 50% of the time
- baselines: LightGBM ~ logistic regression; 1 s macro-F1 0.54, directional accuracy 0.53, IC 0.05; signal gone by 5-10 s
- NN: MLP small (4.5k params) and large (413k) on the same features; all models tie (1 s macro-F1 0.53-0.54),
  large MLP overfits after 1-2 epochs -> gate 1 fails on this label; write-up in results/baseline_results.md.
  Next: event-time (next mid change) label, DeepLOB on FI-2010
- event-time target (direction of next mid change, rows at best bid/ask updates): accuracy 0.66 (LightGBM),
  0.657 (MLP), 0.638 (logistic) vs 0.626 for a one-feature rule; old 1 s target flat at ~0.53 macro-F1.
- learning curve 1->3 training days: +0.2-0.4 pts on the new target, nothing on the old -> target matters far
  more than data; model size still irrelevant. Fixed a macOS crash (torch + lightgbm libomp clash in one process).
- design decision: teacher and student get the same information (teacher = raw event window, student = summary
  features of the same window); cross-asset features go to both; privileged-info teacher only as a side ablation.
  Next: DeepLOB on FI-2010, DeepLOB on event-time ITCH, cross-asset features for all models, distillation, latency

16/11: DeepLOB reproduction on FI-2010 Setup 2 (Table II): weighted F1 82.1 / 72.1 / 78.3 vs paper 83.4 / 72.8 / 80.4
  (k = 10 / 20 / 50), within 0.7-2.1 pts. Paper-size model (61.6k params), paper hyper-parameters, blind test days 8-10.
- latency: DeepLOB 1.8-2.1 ms batch-1 on CPU vs 0.19 ms per sample batched on GPU (paper quotes 0.253 ms); student 0.01 ms.

16/11 (evening): latency simulator (code/sim/latency_sim.py) + saved test predictions; first run: at 10 us
  infrastructure the student (10 us inference) beats LightGBM (27 us) 455 vs 424 ticks per 1,000 decisions; with a
  50 us rival, value is all-or-nothing on being faster. Project framing sharpened: value of speed depends on the
  competitive environment, so results are curves vs total latency with and without a rival.
- Nasdaq DeepLOB, FI-2010 seeds 1-2 and z-score runs queued overnight (code/run_overnight.sh).

16/11 (night): DeepLOB on Nasdaq test acc 0.663 (= LightGBM, +0.66 pts over small MLP); charged 1.76 ms it saves
  228 vs 455 ticks/1k decisions. Distillation: small MLP + LightGBM soft labels 0.657 -> 0.662 (3 seeds), best CPU
  model in the simulator up to ~1 ms; DeepLOB teacher helps little (its edge is not expressible from the student's
  features). Next: student with raw-window inputs to carry DeepLOB's edge; FI-2010 seeds and z-score overnight.

17/11: raw-window students (last 10 updates x top 5 levels + features): distilled tiny conv (8.4k params) 0.665
  test acc > LightGBM 0.663 = DeepLOB 0.663; with raw inputs the DeepLOB+LightGBM teacher becomes the best one. In
  PyTorch its 22 us costs more than the gain at co-located speeds; C implementation next. Feature count corrected
  to 34. z-score FI-2010 files downloaded and verified; FI-2010 seed 1: F1 81.9 / 73.0 (k=10/20).

18/11: FI-2010 seeds 1-2 done: weighted F1 mean 82.08 / 72.38 / 78.62 (3 seeds) vs paper 83.40 / 72.82 / 80.35.
  z-score files: 82.34 (k=10), 72.88 (k=20, matches paper); k=50 z-score run interrupted at epoch 21, to re-run.
