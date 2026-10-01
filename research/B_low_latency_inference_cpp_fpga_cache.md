# Topic B: Ultra-low-latency ML inference engineering (C++, cache, FPGA, kernel bypass)

Verification key: V = fetched and confirmed on the page; S = seen only in search snippet; A = vendor/anecdotal; N = not found. I did not find hard ns-level benchmarks for hand-rolled C++ MLPs; those figures below are estimates and are marked as such.

## Summary (key takeaways)

- FPGA inference is the only approach with published, audited sub-10 us numbers for recurrent nets: Myrtle.ai VOLLO on STAC-ML reports 99p latency of about 5.1 us for the smallest LSTM (LSTM_A), and about 24 us in the earlier Agilex run (S, STAC / press).
- A tiny MLP on FPGA via hls4ml reached "on the scale of 100 ns" (V, [arXiv 1804.06913](https://arxiv.org/abs/1804.06913)). A small CNN reached about 5 us (S).
- On GPU, the same STAC LSTM_A was 54.1 us on an A100 (S). CPU inference in a vendor comparison was 26-163 us with spikes versus about 1.1-1.2 us median for FPGA (V, vendor ICC/Xelera, treat as marketing).
- Software C++ numbers for a tiny MLP were not found in a citable benchmark. From first principles (see below), a 2-3 layer int8 MLP of 10-50k parameters should run in the low hundreds of ns to a few us on one modern core. This is an estimate, not measured.
- A model whose weights fit in L1d (32-48 KB) means no cache misses after warm-up. Zen 4 L1D is 4 cycles (about 0.7 ns), L2 is 1 MB at 14 cycles (V, [Chips and Cheese](https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion)).
- Network/kernel path matters as much as the model: kernel bypass (Onload about 1-2 us, ef_vi sub-us) is a precondition for the model's nanoseconds to matter (S, A).

## Detailed findings

### 1. C++ inference without PyTorch
- Standard routes: hand-written matvec with AVX2/AVX-512 intrinsics, Eigen fixed-size matrices (compile-time sizes allow full unrolling), ONNX Runtime (C++ API, has per-call overhead of microseconds, so likely unsuitable for sub-us budgets; not verified by me), TVM (AOT compilation to C), and code-generating the weights as `constexpr` arrays.
- Trees: [Treelite](https://treelite.readthedocs.io/en/0.31/benchmark.html) compiles XGBoost/LightGBM/sklearn ensembles to C shared libraries; its benchmark pages report 2-6x higher throughput than native XGBoost prediction (V by search summary), but throughput not per-sample latency, so no latency figure was found.
- Batch-1 matters: int8 VNNI FMAs give about 4x the FP32 FMA throughput, and matrix-vector (batch-1) is memory-bound, so keeping weights resident in cache is the main lever ([arXiv 2508.06753](https://arxiv.org/pdf/2508.06753), [OpenVINO int8 doc](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_Int8Inference.html), S).
- Rough estimate (mine, unverified): a 32 B/cycle load pipe at about 5 GHz bounds streaming 40 KB of weights to about 250-300 cycles (about 50-60 ns) from L1, plus compute and dependency stalls between layers, so around 100-500 ns for a 3-layer net. Needs benchmarking on our hardware.

### 2. Cache fit
- Zen 4: L1D 32 KB (from my knowledge; not shown in fetched text), 4-cycle latency; L2 1 MB, 14 cycles. Golden Cove: L1D 48 KB (my knowledge; a search snippet listed "80 KB" which is the combined 32 KB I + 48 KB D), L1D 5 cycles, L2 1.28 MB, 15 cycles (V for latencies, [Chips and Cheese](https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion)). Apple M-series L1D is 128 KB (S, unverified).
- Capacity arithmetic: 32 KB L1d holds 32k int8 weights or 64k int4 weights, minus space for activations, input features and the rest of the trading loop. L2 (1 MB) holds about 1M int8 params at about 14 cycles per access.
- In a real trading process other code evicts L1; pin the core, isolate it, and keep the model plus state in a small hot region. Prefetch the next layer.

### 3. FPGA/ASIC
- hls4ml: jet-substructure MLP, latency on the scale of 100 ns, fits modern FPGAs (V, [arXiv 1804.06913](https://arxiv.org/abs/1804.06913)). Physics-trigger context, not trading. CNN about 5 us ([Chalmers record](https://research.chalmers.se/en/publication/525350), S).
- STAC-ML (the only audited finance-specific ML latency benchmark I found): Myrtle.ai VOLLO on Intel Agilex, LSTM_A 99p 24.0-24.1 us in the first report; 5.1 us in the later Tacana suite; A100 GPU 54.1 us for 16 model instances. Sources: [STAC VOLLO report](https://docs.stacresearch.com/node/47398), [Tacana report](https://stacresearch.com/node/47622), [Intel brief](https://intel.com/content/dam/www/central-libraries/us/en/documents/2024-07/agilex-and-myrtle-vollo-on-stac-ml-performance-brief.pdf), [Myrtle press](https://jotup.co/node/2421577). I could not open the STAC pages directly (404 on fetch), so figures are from snippets (S).
- Vendor: ICC + Xelera Silva claims 1.13-1.24 us median, p99 below 1.4 us, versus CPU 26-163 us ([ICC](https://www.icc-usa.com/icc-xelera-ultra-fast-ml-inference-for-trading), V that the page says this; A as a vendor claim, model unspecified).
- AMD Alveo UL3524: under 3 ns FPGA transceiver latency, 7x better than the previous generation; AMD pushes FINN (quantized PyTorch to hardware IP) for AI ([AMD](https://ir.amd.com/news-events/press-releases/detail/1158/amd-unveils-purpose-built-fpga-based-accelerator-for), V). This is transceiver latency, not model latency.
- FPGA XGBoost for LOB mid-price: a HKUST thesis claims deterministic sub-us end-to-end ([HKUST](https://researchportal.hkust.edu.hk/en/studentTheses/fpga-based-acceleration-for-high-frequency-trading/), S; thesis, not peer-reviewed). Also [FINN-GL (arXiv 2506.20810)](https://arxiv.org/pdf/2506.20810) for FPGA LSTMs (contents not verified).
- ASIC: Rebellions LightTrader/ATOM (CGRA, 64 TOPS int8 in LightTrader, ATOM 128 INT8 TOPS at 150 W) ([HPCA'23 slides](https://rebellions.ai/wp-content/uploads/2023/11/RebellionsIONHPCA23_LightTrader.pdf), S, vendor). No latency per query verified.
- Exegy, Enyx, Cisco/Nexus: searched; no ML inference latency figures found (N).
- LOB-specific: [arXiv 2606.25986](https://arxiv.org/abs/2606.25986) proposes FastBiNLOB, an axis-separable mixer with hardware-friendly ops that is more accurate at lower latency than DeepLOB-type SOTA on FI-2010; abstract confirmed, but I could not extract numeric latencies or parameter counts (not read in full). It also notes most LOB papers report no deployment latency.

### 4. GPU vs CPU vs FPGA at batch 1
- XGBoost, batch 1: FPGA 460 us (including PCIe round trip), GPU 833 us, CPU 7 ms ([arXiv 2110.11719](https://arxiv.org/pdf/2110.11719), snippet S; PDF unreadable to me). PCIe round-trip dominates; this is a data-centre setup, not a trading one.
- V100 TensorRT batch 1 best latency 110 us (S, from [FaaST/NVIDIA material](https://arxiv.org/pdf/2010.08556)). GPUs launch/transfer overhead puts floors in tens of us. Conclusion: GPU is for training and teacher models, not the hot path.

### 5. Kernel bypass
- Onload (LD_PRELOAD shim) about 1-2 us; ef_vi sub-us ([techinterview blog](https://www.techinterview.org/post/3233477282/hft-kernel-bypass-interview/?format=md), A). Solarflare FPGA-based tick-to-trade claims: 98 ns max actionable latency, about 22 ns network portion ([Finextra](https://www.finextra.com/pressarticle/69101/solarflare-slashes-tick-to-trade-latency), [STAC-T0](https://stacresearch.com/node/43535), A/vendor). DPDK: no numbers gathered.
- Implication: with bypass, the software decision path is a tight poll loop on one core; the model must fit that loop.

## Table

| Technique | Latency / size figure | Source | Reliability |
|---|---|---|---|
| hls4ml MLP on FPGA | about 100 ns | [arXiv 1804.06913](https://arxiv.org/abs/1804.06913) | High (peer-reviewed, physics task) |
| hls4ml CNN | about 5 us | [Chalmers](https://research.chalmers.se/en/publication/525350) | Medium (snippet) |
| VOLLO LSTM_A, FPGA | 5.1 us p99 (later), 24 us p99 (earlier) | [STAC](https://stacresearch.com/node/47622) | Medium-high (audited, but not read directly) |
| LSTM_A on A100 | 54.1 us | [STAC](https://docs.stacresearch.com/node/47398) | Medium |
| FPGA ML (Xelera) | 1.13-1.24 us median | [ICC](https://www.icc-usa.com/icc-xelera-ultra-fast-ml-inference-for-trading) | Low (vendor) |
| CPU baseline in above | 26-163 us | same | Low (vendor, unspecified model) |
| UL3524 transceiver | under 3 ns | [AMD](https://ir.amd.com/news-events/press-releases/detail/1158/amd-unveils-purpose-built-fpga-based-accelerator-for) | Vendor, not model latency |
| Treelite vs XGBoost | 2-6x throughput | [Treelite](https://treelite.readthedocs.io/en/0.31/benchmark.html) | Medium; no latency |
| XGBoost batch 1 FPGA/GPU/CPU | 460 us / 833 us / 7 ms | [arXiv 2110.11719](https://arxiv.org/pdf/2110.11719) | Medium (snippet); PCIe-bound |
| Zen 4 L1D / L2 | 4 / 14 cycles; L2 1 MB | [Chips and Cheese](https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion) | High |
| Golden Cove L1D / L2 | 5 / 15 cycles; L2 1.28 MB | same | High |
| Onload / ef_vi | 1-2 us / sub-us | [techinterview](https://www.techinterview.org/post/3233477282/hft-kernel-bypass-interview/?format=md) | Low (blog) |
| Hand C++ int8 MLP (about 40 KB) | about 0.1-0.5 us | my estimate | Unverified; must measure |

## Practical implications for the student model

- Parameter budget: target at most 30k int8 params (about 30 KB) to fit in a 32 KB L1D with room for activations; up to about 48k on Golden Cove-class L1D. int4 doubles this (about 60k) but needs unpack cost and accuracy checks. A relaxed tier: up to about 500k int8 params resident in L2 (about 1 MB), at some ns cost per layer.
- Architecture: MLP or a small 1-D temporal-conv or gated tree/MLP over engineered LOB features. LSTMs have sequential dependencies and are hard on CPU (but FPGA handles them: STAC). Trees (LightGBM via Treelite or hand-compiled) are a strong baseline student; branch mispredictions are the risk.
- Quantization: int8 weights + int8 activations (per-channel scales), QAT during distillation; keep the final logit in int32/float. int4 only as an ablation. Use power-of-two scales so dequant is a shift.
- Layout: row-major weights padded to SIMD width, layers contiguous, 64-byte aligned, no heap allocation in the hot path, `-O3 -march=native`, pinned isolated core.
- Report in the paper: parameter count, bytes, and a measured p50/p99/p99.9 ns single-sample latency from a C++ harness (rdtsc), alongside accuracy versus teacher. This is our contribution, since the literature rarely reports it.

## Open questions
- Real measured ns for 10k-100k param int8 MLPs on our CPU (Apple Silicon NEON vs x86 AVX2/VNNI); need our own benchmark.
- ONNX Runtime per-call overhead at batch 1 (not verified).
- STAC-ML model sizes for LSTM_A (parameter count) to compare with our student (could not open page).
- Full-text numbers for FastBiNLOB (arXiv 2606.25986) and the XGBoost-FPGA HKUST thesis.
- Whether to do FPGA (hls4ml/FINN) as a stretch goal or just cite.

## Sources
- hls4ml: https://arxiv.org/abs/1804.06913
- Fast CNNs with hls4ml: https://research.chalmers.se/en/publication/525350
- STAC VOLLO: https://docs.stacresearch.com/node/47398 ; Tacana: https://stacresearch.com/node/47622 ; STAC news: https://stacresearch.com/news/testing-the-limits-of-machine-learning-inference-in-finance
- Intel/Myrtle brief: https://intel.com/content/dam/www/central-libraries/us/en/documents/2024-07/agilex-and-myrtle-vollo-on-stac-ml-performance-brief.pdf
- Myrtle press: https://jotup.co/node/2421577
- ICC/Xelera: https://www.icc-usa.com/icc-xelera-ultra-fast-ml-inference-for-trading
- AMD UL3524: https://ir.amd.com/news-events/press-releases/detail/1158/amd-unveils-purpose-built-fpga-based-accelerator-for
- FINN-GL: https://arxiv.org/pdf/2506.20810
- FastBiNLOB: https://arxiv.org/abs/2606.25986
- HKUST FPGA thesis: https://researchportal.hkust.edu.hk/en/studentTheses/fpga-based-acceleration-for-high-frequency-trading/
- Rebellions LightTrader: https://rebellions.ai/wp-content/uploads/2023/11/RebellionsIONHPCA23_LightTrader.pdf
- Treelite benchmark: https://treelite.readthedocs.io/en/0.31/benchmark.html
- PCIe streaming FPGA: https://arxiv.org/pdf/2110.11719 ; FaaST: https://arxiv.org/pdf/2010.08556
- Zen 4 memory: https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion
- Int8 CPU: https://arxiv.org/pdf/2508.06753 ; https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_Int8Inference.html
- Kernel bypass: https://www.techinterview.org/post/3233477282/hft-kernel-bypass-interview/?format=md ; https://www.finextra.com/pressarticle/69101/solarflare-slashes-tick-to-trade-latency ; https://stacresearch.com/node/43535
