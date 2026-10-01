# Topic C: Distillation and Post-Training / Compression Methods

Verification note: items marked [V] were opened/confirmed during this research (page fetched or search hit). Items marked [ID] have an arXiv ID I am confident of from memory but did not re-open; spot-check before citing in the report. Nothing here is knowingly invented.

## 1. Summary

- For an HFT market-making student (microsecond to low-millisecond budget), the practical recipe is: train a large teacher offline, distil into a tiny MLP / small GBDT / shallow tree / small 1D-CNN or GRU on engineered LOB features, then quantize (int8) and optionally prune. Output distillation (soft targets or teacher regression outputs) is the baseline; feature/hint losses and tree/LUT students are the main variations worth ablating.
- Distillation is most reliable when the teacher is much better than a student trained on hard labels AND the student has enough capacity. On noisy, low-SNR financial targets the teacher's soft outputs act as a denoised label, which is the main plausible gain. Evidence that this generalises is thin (a handful of small papers).
- LLM distillation (sequence-level, on-policy, reasoning-trace SFT) is a different regime: huge data, generative tokens, teacher sampling. Only the core ideas (soft targets, on-policy student-state training) carry over to a tiny tabular/time-series student.
- Frontier labs: it is confirmed that Google (Gemini Flash, Gemma) and Anthropic (via its own statement) say distillation of own models into smaller ones is routine; OpenAI sells a distillation product. Specific recipes for specific Claude/GPT models are not public.
- Known limits: capacity gap, teacher-student fidelity gap even when capacity suffices, and little benefit if teacher is not much better than the student.

## 2. Taxonomy of methods

### 2.1 Distillation families
- **Classic logit/soft-target KD** (Hinton, Vinyals, Dean 2015): student matches temperature-softened teacher outputs plus hard-label loss. [V-ID] https://arxiv.org/abs/1503.02531
- **Hint / feature distillation (FitNets)**: regress a student intermediate layer onto a teacher hidden layer via a learned projector. https://arxiv.org/abs/1412.6550 [ID]
- **Attention transfer** (Zagoruyko & Komodakis): match spatial attention maps; for us, analogue is matching teacher attention/saliency over time steps or features. https://arxiv.org/abs/1612.03928 [ID]
- **Sequence-level KD** (Kim & Rush 2016): train on teacher-decoded outputs rather than distributions. https://arxiv.org/abs/1606.07947 [ID]
- **Self-distillation / born-again networks**: student has same architecture as teacher and still improves; iterate generations. Born-Again Networks https://arxiv.org/abs/1805.04770 [ID]; Be Your Own Teacher (deep supervision of shallow heads) https://arxiv.org/abs/1905.08094 [ID]
- **Online / mutual distillation**: peers train together, no pre-trained teacher. Deep Mutual Learning https://arxiv.org/abs/1706.00384 [ID]. Applied to finance: online KD for financial time series (INISTA 2022, Aristotle Univ. Thessaloniki) https://cidl.csd.auth.gr/resources/conference_pdfs/Online%20Knowledge%20Distillation%20for%20Financial%20Timeseries%20Forecasting_INISTA2022.pdf [V search hit]
- **Teacher-assistant KD** for large gaps: intermediate-size model bridges teacher and student. Mirzadeh et al. https://arxiv.org/abs/1902.03393 [V]
- **Patient and consistent teacher** (same inputs/augmentations for teacher and student, long training): https://arxiv.org/abs/2106.05237 [ID]

### 2.2 Distilling into trees / lookup tables
- **Soft decision trees** (Frosst & Hinton): distil a net into a hierarchical mixture of experts tree. https://arxiv.org/abs/1711.09784 [ID]
- **Born-again tree ensembles** (Vidal & Schiffer): turn a tree ensemble into a single tree. https://arxiv.org/abs/2003.11132 [ID]
- **NODE** (neural oblivious decision ensembles): differentiable oblivious trees, tabular-friendly, fast inference. https://arxiv.org/abs/1909.06312 [ID]
- **Generic approach**: fit a GBDT / shallow tree / binned lookup table on teacher soft outputs (and teacher-labelled synthetic data). Inference is a few comparisons or one table lookup, ideal for nanosecond to microsecond budgets; FPGA-friendly. (Standard practice; I did not find a single canonical paper.)

### 2.3 Early exit / conditional compute
- **BranchyNet**: side-branch classifiers exit when entropy is low. https://arxiv.org/abs/1709.01686 [ID]
- For MM: cheap model handles easy states, escalate to the larger model when uncertainty is high; also a natural "cascade" design.

### 2.4 Pruning
- **Magnitude pruning + Deep Compression** (Han et al.): prune, quantize, Huffman-code. https://arxiv.org/abs/1510.00149 [ID]
- **Lottery ticket hypothesis** (Frankle & Carbin): sparse subnetworks trainable from early init. https://arxiv.org/abs/1803.03635 [ID]
- Caveat: unstructured sparsity rarely speeds up CPU/GPU inference for tiny dense nets; structured pruning (neurons/channels) is what reduces latency.

### 2.5 Quantization
- **PTQ vs QAT**: PTQ calibrates scales on a small data set (fast, can lose accuracy at int4); QAT simulates quantization during training (better at low bit width). Jacob et al. integer-only inference https://arxiv.org/abs/1712.05877 [ID]; Krishnamoorthi whitepaper https://arxiv.org/abs/1806.08342 [ID].
- int8 is usually near-lossless for MLPs/CNNs; int4 usually needs QAT. Tiny nets may not be memory-bound, so gains come mainly from integer SIMD and cache residency.

### 2.6 Low-rank and architecture search
- **Low-rank factorization** of weight matrices (Denton et al.): https://arxiv.org/abs/1404.0736 [ID]. For tiny MLPs the gain is small; matters more for wide layers.
- **NAS**: Zoph & Le https://arxiv.org/abs/1611.01578 [ID]; Once-for-All (train once, specialise subnets to latency targets) https://arxiv.org/abs/1908.09791 [ID]. Latency-aware NAS with a hardware-measured budget is the relevant variant; a plain grid over width/depth is likely enough for our scale.

## 3. LLM vs traditional DL distillation

| Aspect | LLM distillation | Traditional (our setting) |
|---|---|---|
| Output | Token distributions over large vocab, generated sequences | Class probs or a few regression targets |
| Data | Huge corpora plus teacher-generated text | Fixed dataset; teacher can label unlimited unlabeled/augmented rows |
| Key issue | Exposure bias; student visits states teacher never would | Capacity, noise, non-stationarity |
| Loss | Forward KL, reverse KL (MiniLLM), generalized JSD (GKD) | KL on soft labels, MSE on regression outputs, hint losses |
| Cost driver | Teacher sampling / logits storage | Teacher forward pass is cheap offline |

Key LLM references:
- DistilBERT https://arxiv.org/abs/1910.01108 [ID]; TinyBERT (layer-wise and attention distillation) https://arxiv.org/abs/1909.10351 [ID]
- MiniLLM (reverse KL, policy-gradient optimisation) https://arxiv.org/abs/2306.08543 [ID]
- On-Policy Distillation of LMs / GKD (student's own samples scored by teacher; ICLR 2024) https://arxiv.org/abs/2306.13649 [V]
- On-policy distillation blog, Thinking Machines Lab, 27 Oct 2025 (per-token reverse KL on student rollouts; claims large compute savings vs RL) https://thinkingmachines.ai/blog/on-policy-distillation/ [V via search; read primary for exact numbers]
- DeepSeek-R1: reasoning traces from R1 used for SFT of smaller Qwen/Llama models (sequence-level distillation, no logits) https://arxiv.org/abs/2501.12948 [ID]
- Gemma 2 (small models trained with distillation from a larger teacher instead of next-token prediction) https://arxiv.org/abs/2408.00118 [ID]; Gemma 3 https://arxiv.org/abs/2503.19786 [ID]
- Distillation scaling laws (Apple; ICML 2025): distillation beats supervised training only up to a compute threshold, and when a teacher already exists or is reused; training teacher just for one student is usually worse than plain supervised. https://arxiv.org/abs/2502.08606 [V]

Transfer to our problem: the on-policy idea maps to "DAgger-style" distillation of a market-making policy: label states the student visits in simulation with the teacher's actions. This matters if the student is a policy (quote/cancel decisions) rather than a pure predictor. Sequence-level/trace distillation has no direct analogue.

## 4. What frontier labs publicly say (confirmed vs speculation)

Confirmed (primary or near-primary):
- Google: Gemini 1.5 Flash is described as online-distilled from the larger Gemini 1.5 Pro in the Gemini 1.5 report https://arxiv.org/abs/2403.05530 [ID for the report; the "online distillation" claim confirmed by secondary summaries in search, check wording in the report]. Gemma 2 uses distillation (above).
- OpenAI: launched a Model Distillation product (Stored Completions plus evals plus fine-tuning) on 1 Oct 2024 https://openai.com/index/api-model-distillation/ [V search hit]. That is a customer feature; it does not by itself state how OpenAI builds its own small models (e.g. mini variants).
- Anthropic: its post "Detecting and preventing distillation attacks" (23 Feb 2026) states that distillation is "a widely used and legitimate training method" and that "frontier AI labs routinely distill their own models to create smaller, cheaper versions for their customers." https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks [V]. It also alleges large-scale illicit distillation of Claude by other labs. This is a general statement; Anthropic does not detail which Claude models are distilled or how.

Speculation (not confirmed): that specific smaller tiers (Haiku, GPT mini, etc.) are produced by logit-level distillation versus sequence-level or from-scratch training with synthetic data. Treat as unknown. Also note model-extraction research showing partial recovery of production model parameters via API access: https://arxiv.org/abs/2403.06634 [ID].

## 5. What is best for tiny tabular / time-series students

Evidence and reasoning:
- **Trees often beat deep nets on tabular data** (Grinsztajn et al., NeurIPS 2022) https://arxiv.org/abs/2207.08815 [ID]. So a GBDT/shallow tree student distilled from a deep teacher, or even a GBDT trained directly, is a strong baseline that must be included.
- **Regression distillation** lacks class-correlation "dark knowledge"; use MSE/Huber on teacher outputs, ideally also distribution outputs (quantiles, mixture params) or multi-horizon / multi-task outputs to give a richer signal. Paper addressing exactly this for stock volume: Distributional Correlation-Aware KD, https://arxiv.org/abs/2208.07232 [V search hit].
- **Financial time-series distillation papers**:
  - Prior knowledge distillation based on financial time series (co-distillation to a smaller structure to cut overfitting from noise) https://arxiv.org/abs/2006.09247 [V search hit]
  - TIPS: multi-teacher distillation of inductive biases into a Transformer for financial forecasting, KDD 2026, claims 38% of typical inference compute https://arxiv.org/abs/2603.16985 [V]
  - Stock trading volume KD above. Few of these target latency-critical LOB/market-making, which is a gap we can claim, but verify by additional search (other topics' researchers may cover LOB models, e.g. DeepLOB https://arxiv.org/abs/1808.03668 [ID]).
- **Unlabeled teacher-labelled data**: because teacher inference is offline, we can label extra simulated or augmented LOB windows, which is the strongest lever against noisy labels.
- **Quantization** is cheap and nearly free for MLP students; apply last.

## 6. Limits and failure modes

- **Capacity gap**: very small student cannot absorb a big teacher; accuracy can drop as teacher gets stronger. Mitigations: teacher assistant [V], earlier-stopped teacher checkpoints, multi-teacher.
- **Fidelity gap even with enough capacity**: students often fail to match the teacher's function, and better matching does not guarantee better generalisation (Stanton et al., "Does Knowledge Distillation Really Work?", NeurIPS 2021) https://arxiv.org/abs/2106.05945 [V]. Also "On the efficacy of knowledge distillation" (Cho & Hariharan) https://arxiv.org/abs/1910.01348 [ID].
- **Compute economics**: if you must train the teacher only for one student, plain supervised training may be as good [V, scaling laws].
- **Noise and non-stationarity**: a teacher that overfits noise or regime-specific patterns passes those on; distillation does not add information absent from the data. Out-of-time evaluation is mandatory.
- **Distribution shift in policy settings**: student acting on its own induced states (inventory, queue position) diverges from teacher-visited states, so use on-policy/DAgger-style labelling or evaluate in a closed-loop simulator, not just offline MSE.
- **Evaluation trap**: matching accuracy or MSE is not PnL. Latency gains can be eaten by adverse selection if the student is slightly worse at the tails; report fill-conditioned metrics.
- **Pruning**: unstructured sparsity often gives no wall-clock win on small nets.

## 7. Recommended ablations for our project

1. Baselines: student trained on hard labels only; GBDT on hard labels; teacher itself (accuracy upper bound and latency lower bound).
2. Distillation signal: hard-only vs soft/regression-only vs mixed (alpha sweep); temperature sweep (classification); with and without extra teacher-labelled synthetic or unlabeled data.
3. Feature/hint distillation vs output-only; attention/saliency transfer if teacher is a Transformer.
4. Student family at matched latency: MLP, small GRU/1D-CNN, shallow tree, GBDT, oblivious trees (NODE), lookup table. Report the accuracy-latency Pareto frontier measured on real hardware (p50/p99/p99.9 latency, single-threaded, batch size 1).
5. Capacity gap: student width/depth sweep; teacher size sweep; teacher-assistant chain.
6. Self-distillation / born-again generations; online mutual distillation among small students.
7. Compression stack: int8 PTQ vs QAT vs int4; structured pruning ratio; low-rank; early-exit cascade with uncertainty threshold, measured end to end.
8. Robustness: out-of-time test split, different regimes (high vol vs calm), different instruments; distillation-induced fidelity (agreement with teacher) vs downstream market-making PnL in a simulator.
9. On-policy (DAgger-style) distillation vs offline-only if the student is a quoting/cancel policy.

## 8. Source list

Foundational: 1503.02531 (Hinton), 1412.6550 (FitNets), 1612.03928 (attention transfer), 1606.07947 (seq-level KD), 1805.04770 (born-again), 1905.08094 (self-distillation), 1706.00384 (DML), 1902.03393 (TAKD), 2106.05237 (patient teacher), 2106.05945 (does KD work), 1910.01348 (efficacy of KD), 2502.08606 (distillation scaling laws). All at https://arxiv.org/abs/<id>.
Trees/tabular: 1711.09784, 2003.11132, 1909.06312, 2207.08815.
Compression: 1709.01686, 1510.00149, 1803.03635, 1712.05877, 1806.08342, 1404.0736, 1611.01578, 1908.09791.
LLM: 1910.01108, 1909.10351, 2306.08543, 2306.13649, 2501.12948, 2408.00118, 2503.19786, 2403.05530, 2403.06634; https://thinkingmachines.ai/blog/on-policy-distillation/
Industry statements: https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks ; https://openai.com/index/api-model-distillation/
Finance: 2006.09247, 2208.07232, 2603.16985, INISTA 2022 online KD paper (link above), 1808.03668 (DeepLOB).
