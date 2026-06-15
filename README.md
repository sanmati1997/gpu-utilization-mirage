# The GPU Utilization Mirage

**A reproducible measurement of why "GPU utilization" overstates how much useful work a GPU is doing, and what an AI infrastructure platform could surface instead.**

[Research write-up](PAPER.md) · [Reproduce on a free Colab T4](#reproduce) · Built by [Sanmati Sawalwade](https://linkedin.com/in/sanmati-sawalwade)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanmati1997/gpu-utilization-mirage/blob/main/notebooks/gpu_utilization_mirage.ipynb)

---

## Research question

When an enterprise AI platform reports rising "GPU utilization," what is actually being measured, and does it track the useful work the GPU performs?

This repository answers that with a small, fully reproducible experiment on a single GPU, and proposes a concrete metric an AI infrastructure platform could surface to close the gap.

## TL;DR

`nvidia-smi` GPU-Util reports the fraction of time a kernel is running. It can read ~100% while the GPU performs a fraction of the math it is capable of. **Model FLOPs Utilization (MFU)** — achieved FLOPs over the GPU's peak FLOPs for the precision in use — measures the useful work. The two diverge sharply, and the divergence is driven by configuration (batch size, precision, input pipeline), not by the hardware.

This matters for **PaletteAI** (Spectro Cloud + NVIDIA, launched Oct 2025), whose headline claim is raising GPU utilization "from ~30% to ~60%." That is an *allocation* metric — it measures fewer idle GPUs. The larger, less visible loss is *work-efficiency inside the jobs that are already running*. The raw signal needed to measure it (DCGM SM-active and tensor-pipe-active counters) already ships in the NVIDIA GPU Operator that Palette automates. It is not yet productized into MFU, dollar-attributed waste, or auto-flagged inefficiency.

**This is product insight, not a novel result.** MFU is established (Google PaLM, Karpathy's nanoGPT) and the "GPU utilization is misleading" point has been made before ([Trainy](https://www.trainy.ai/blog/gpu-utilization-misleading)). The contribution here is (a) a clean, honest, reproducible measurement of the gap, and (b) a concrete proposal for the management-plane surface that would expose it.

## Findings (Tesla T4 reference run)

Three results. The memory-bound figure is exact; the training-MFU absolutes are from one T4 run (reproduce with the [notebook](notebooks/gpu_utilization_mirage.ipynb)), and the **ratio — how much MFU moves — is the robust takeaway**, independent of any single peak-FLOPs choice.

**1. The metric gap is real (microbenchmark).** A memory-bound elementwise op: **100% GPU-Util, 0.1% MFU.** The dashboard reads "maxed out"; the GPU does almost no useful math.

**2. A real GPT training step (not a synthetic matmul).** At batch 8 on a T4: **GPU-Util ~96% while MFU was ~12%.** It looks busy while doing roughly an eighth of its useful work.

**3. The gap is fixable by config, not hardware.** Scaling batch size from 1 to 48 lifted MFU **~8×** (≈2% → ≈16%) on the same GPU — GPU-Util sat near 100% the entire time.

A **roofline** (in the notebook) shows the microbenchmark gap is *principled*, not a fluke: the memory-bound op sits on the bandwidth roof, the big matmul right of the ridge. GPU-Util can't tell you which side of the ridge you're on; the roofline can.

> Honesty notes: MFU is computed against the **dense, precision-matched** peak using **non-embedding** parameter count (token/position embeddings excluded — they do no matmul, and counting them inflates MFU ~1.24× for this config). The "8×" uses batch=1 as the floor (an extreme baseline; it illustrates the shape — small effective batches are common with large models or tight memory). Absolute MFU varies by GPU and run; reproduce via the notebook.

| Metric | What it measures | Typical reading |
| --- | --- | --- |
| `nvidia-smi` GPU-Util | a kernel is executing | near 100% |
| DCGM `SM_ACTIVE` | an SM has ≥1 warp resident | high, forgiving |
| DCGM `PIPE_TENSOR_ACTIVE` | tensor-core cycles doing matmul | lower, sharper |
| **MFU (computed)** | useful FLOPs ÷ GPU peak FLOPs | the truth |

## Why it matters in dollars

A job that holds a full GPU at "100% utilization" but runs at low MFU is paying for compute it does not use. The cost model in [`analysis/dollar_model.py`](analysis/dollar_model.py) reports two figures, deliberately:

- `headroom_vs_peak` — gap to the theoretical ceiling. Aspirational, not reclaimable.
- `recoverable` — the same work retuned to an achievable MFU target (default 45%, near PaLM's reported 46%). **This is the defensible number.** Lead with it.

---

## Reproduce

**Fastest path — one click:** open [`notebooks/gpu_utilization_mirage.ipynb`](notebooks/gpu_utilization_mirage.ipynb) in Colab (badge at top), set runtime to GPU (free T4), run top to bottom (~15 min). It covers the microbenchmarks, the roofline, the real GPT training step, and the batch-size sweep, and saves the charts.

The scripts below are the same logic as standalone CLIs. The headline (GPU-Util vs MFU) runs on a **free Colab T4**. The DCGM middle rung needs a host where you are root (a rented A10/L4/A100 VM).

```bash
pip install -r requirements.txt

# Phase 1 — the headline finding (GPU-Util vs MFU), free Colab T4
python measure/train_loop.py --steps 200 --batch-size 8 --seq-len 256

# Phase 2 — prove the gap is fixable by configuration, not hardware
python measure/sweep.py --mode batch        # batch size is the #1 lever
python measure/sweep.py --mode precision     # fp16/bf16 vs fp32
python measure/sweep.py --mode dataloader    # input starvation

# Phase 3 — translate to dollars (edit jobs with your measured MFU)
python analysis/dollar_model.py

# Phase 4 — the proposed PaletteAI surface (local, not Colab)
streamlit run panel/app.py

# DCGM middle rung — on a VM you control, two terminals:
#   A: python measure/train_loop.py --steps 100000 --batch-size 8
#   B: python measure/dcgm_capture.py --seconds 20
```

### Colab quickstart

```python
!nvidia-smi -L
!git clone https://github.com/sanmati1997/gpu-utilization-mirage
%cd gpu-utilization-mirage
!pip -q install -r requirements.txt
!python measure/train_loop.py --steps 200 --batch-size 8
!python measure/sweep.py --mode batch
```

---

## Method (summary)

A compact decoder-only transformer (nanoGPT-shaped, ~30M non-embedding params) is trained for a fixed number of steps on one GPU. During the timed window:

- GPU-Util is sampled from `nvidia-smi` in a background thread.
- Throughput (tokens/sec) is measured directly.
- Achieved FLOPs = `flops_per_token × tokens_per_sec`, where
  `flops_per_token = 6N + 12 · n_layer · n_embd · seq_len`
  (the standard PaLM/Karpathy approximation plus the attention term).
- MFU = achieved FLOPs ÷ **dense** peak FLOPs for the **precision actually used** (table in [`measure/gpu_peaks.py`](measure/gpu_peaks.py)).

Full methodology, design decisions, and corrections are in [PAPER.md](PAPER.md).

## Threats to validity (read before citing a number)

- **MFU denominator must match training precision and be the dense (non-sparsity) peak.** Dividing fp16 throughput by an fp32 peak, or by a 2:4-sparsity peak, produces large errors. `gpu_peaks.py` stores verified dense peaks and refuses unknown GPU/precision pairs.
- **Weight tying.** The model ties the token embedding and output head (GPT-2 standard). Without it, the token-embedding lookup is falsely counted as matmul work and inflates MFU by ~1.6× for this config.
- **DCGM profiling counters need the host engine and usually root** — they do not run on stock Colab, which is why the headline is intentionally DCGM-free.
- **Single GPU, synthetic data.** Results are directional and meant to demonstrate the metric gap, not to benchmark any specific production workload.
- **AMP mixed precision** runs matmuls at fp16/bf16 and some ops at fp32; MFU is computed against the tensor (matmul) peak, since matmuls dominate FLOPs.

## Related work / prior art (honest)

- **MFU** — Chowdhery et al., *PaLM* (2022); Karpathy, *nanoGPT* `estimate_mfu`.
- **"GPU utilization is misleading"** — Trainy; Modal; multiple SM-efficiency write-ups.
- **Allocation-side GPU sharing** — NVIDIA Run:ai / KAI Scheduler (fractional GPU, SLA-aware), Gandiva, Tiresias, Gavel. This is the layer PaletteAI's 30%→60% number lives in; this repo deliberately addresses the *work-efficiency* layer below it.

## Layout

```
gpu-utilization-mirage/
├── README.md              # this file
├── PAPER.md               # full research write-up
├── requirements.txt
├── notebooks/
│   └── gpu_utilization_mirage.ipynb  # one-click Colab: micro + roofline + real GPT + batch sweep
├── measure/
│   ├── gpu_peaks.py        # verified dense peak-FLOPS table (the MFU denominator)
│   ├── train_loop.py       # workload + GPU-Util sampler + MFU + headline chart
│   ├── sweep.py            # batch / precision / dataloader sensitivity sweeps
│   └── dcgm_capture.py     # DCGM SM-active / tensor-active (host you control)
├── analysis/
│   └── dollar_model.py     # headroom vs recoverable waste
├── panel/
│   └── app.py              # proposed PaletteAI work-efficiency surface (Streamlit)
└── assets/                 # generated charts + json (after you run)
```

---

Built by [Sanmati Sawalwade](https://linkedin.com/in/sanmati-sawalwade) — MS Information Systems, Northeastern University (Silicon Valley)
sawalwade.s@northeastern.edu · [sanmati1997.github.io](https://sanmati1997.github.io)
