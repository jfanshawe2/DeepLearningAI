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
