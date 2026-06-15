# The GPU Utilization Mirage

### A reproducible measurement of the gap between reported GPU utilization and useful work, with a productization proposal for AI infrastructure platforms

**Sanmati Sawalwade** · MS Information Systems, Northeastern University (Silicon Valley)
sawalwade.s@northeastern.edu · [github.com/sanmati1997](https://github.com/sanmati1997)

---

## Abstract

Enterprise AI infrastructure platforms increasingly market "GPU utilization" as a headline efficiency metric. We show, with a small reproducible experiment on a single GPU, that the commonly reported utilization signal (`nvidia-smi` GPU-Util) measures only whether a kernel is executing and systematically overstates how much useful computation a GPU performs. Using Model FLOPs Utilization (MFU) as the reference for useful work, we demonstrate that the two metrics diverge substantially and that the divergence is governed by configuration — batch size, numeric precision, and the input pipeline — rather than by the hardware. This matters for the broad class of GPU platforms built on NVIDIA's GPU Operator, which surface DCGM's metrics: a headline "GPU utilization" gain is typically an *allocation* improvement (fewer idle GPUs), orthogonal to *work-efficiency* inside running jobs. The raw counters required to measure work-efficiency (NVIDIA DCGM SM-active and tensor-pipe-active) already ship in the GPU Operator these platforms build on, but are rarely surfaced as MFU, dollar-attributed waste, or auto-flagged inefficiency. We contribute (1) a clean, honest, reproducible measurement of the metric gap and (2) a concrete proposal and working mockup for the management-plane surface that would expose it. We are explicit that MFU and the "utilization is misleading" observation are established; the contribution is the measurement discipline and the product framing.

---

## 1. Introduction

GPUs are the scarcest and most expensive resource in modern AI infrastructure. It is therefore natural that platform vendors quantify their value in terms of "GPU utilization." The implicit promise is that higher utilization means more useful work per dollar of hardware.

That promise depends entirely on what "utilization" measures. The metric almost universally reported — `nvidia-smi`'s GPU-Util — is defined by NVIDIA as the fraction of time over the past sampling window during which at least one kernel was executing. It says nothing about whether that kernel saturated the compute units, used the tensor cores, or stalled waiting for data. A kernel that occupies the GPU while doing a small amount of arithmetic registers as fully utilized.

This paper asks a deliberately narrow question: **when a platform reports rising GPU utilization, does that track the useful work performed?** We answer it empirically and then translate the answer into a product recommendation.

The motivating context is the wave of GPU platforms — neoclouds, Kubernetes GPU managers, and managed training services — that market rising "GPU utilization" as a measure of value. A representative claim is moving utilization from roughly 30% to 60%. We take such claims at face value and argue they are *additive* to, not contradicted by, the work-efficiency story: a 30%→60% figure is about reducing idle and hoarded GPUs (an allocation/scheduling win), whereas the efficiency of the jobs that *are* running is a separate axis the headline metric does not capture.

## 2. Background

### 2.1 GPU-Util

`nvidia-smi` GPU-Util is a coarse, time-based occupancy signal. It is cheap to read and available everywhere, which is why it dominates dashboards. Its weakness is well documented: it can report 100% while the streaming multiprocessors (SMs) are mostly idle and the tensor cores untouched.

### 2.2 SM activity and tensor-pipe activity (DCGM)

NVIDIA's Data Center GPU Manager (DCGM) exposes profiling fields that are closer to the truth:

- `DCGM_FI_PROF_SM_ACTIVE` — fraction of time at least one warp is resident on an SM. Better than GPU-Util, but still forgiving: one resident warp suffices.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` — fraction of cycles the tensor pipe is active. For matmul-bound training and inference, this is the sharpest readily available "is it doing the real work" signal.

DCGM ships in the NVIDIA GPU Operator. Any platform that automates that operator is already collecting this signal.

### 2.3 Model FLOPs Utilization (MFU)

MFU, introduced in Google's PaLM report and popularized by Karpathy's nanoGPT, is the ratio of achieved throughput to the hardware's theoretical peak:

```
MFU = achieved_FLOPs_per_second / peak_FLOPs_per_second
```

It is hardware-grounded and precision-aware. Well-optimized large-model training typically lands around 35–50% MFU (PaLM reported 46.2%). MFU requires knowing the model's FLOPs per token, which is why it is not a drop-in counter — but for a known architecture it is straightforward to compute.

## 3. Method

### 3.1 Workload

We train a compact decoder-only transformer (nanoGPT-shaped: 6 layers, 6 heads, embedding dim 384, sequence length 256, vocab 50,304; ~30M non-embedding parameters after weight tying) for a fixed number of optimizer steps on a single GPU, after a warmup that lets clocks and caches settle. Inputs are synthetic token tensors; the goal is to characterize the *metric*, not a production dataset.

### 3.2 Measured quantities

During the timed window:

- **GPU-Util** is polled from `nvidia-smi` at 10 Hz in a background thread and averaged over the window.
- **Throughput** is `batch_size × seq_len × steps / elapsed` tokens per second.
- **Achieved FLOPs** is `flops_per_token × tokens_per_second`, where

  ```
  flops_per_token = 6N + 12 · n_layer · n_embd · seq_len
  ```

  `6N` is the standard forward+backward approximation with `N` the non-embedding parameter count; the second term accounts for attention, which the bare `6N` term omits and which matters at longer sequence lengths. (`--no-attention-flops` disables it for comparison.)
- **MFU** is achieved FLOPs divided by the **dense** peak FLOPs for the **precision actually used**, from a verified per-GPU table.

On a host we control, `dcgm_capture.py` additionally samples `SM_ACTIVE` and `PIPE_TENSOR_ACTIVE` during a sustained run, producing the middle rungs of the ladder.

### 3.3 Sensitivity sweeps

To establish that the gap is a *configuration* property and therefore actionable, we hold the model fixed and vary one knob at a time: batch size, numeric precision, and DataLoader worker count (with a deliberately CPU-bound dataset to expose input starvation).

### 3.4 Cost model

We convert MFU into money with two clearly separated figures:

- `headroom_vs_peak = spend × (1 − MFU)` — the gap to the theoretical ceiling. Reported for context only; it is not reclaimable because no real job reaches 100% MFU.
- `recoverable = spend × max(0, 1 − MFU / MFU_target)` — the same work retuned to an achievable target (default 45%), finishing in `MFU / MFU_target` of the time. This is the defensible number and the one we lead with.

## 4. Methodological corrections

Four decisions materially affect correctness and are recorded here because getting them wrong is the common way this kind of measurement misleads:

1. **Precision-matched, dense denominator.** MFU must divide by the peak for the precision in use. NVIDIA datasheets typically headline the 2:4-sparsity peak; the correct denominator for dense training is half that. `gpu_peaks.py` stores verified dense peaks (e.g., L4 fp16 = 121 TFLOPS, not the datasheet's 242) and raises on unknown GPU/precision pairs rather than guessing.

2. **Weight tying.** The token embedding and output head share one matrix (GPT-2 standard). Untied, the token-embedding lookup — which performs essentially no FLOPs — is counted as matmul work under the `6N` convention, inflating MFU by ~1.6× for this configuration (N would be 49M instead of the correct 30M). Tying restores honest accounting.

3. **bf16 only on Ampere and later.** Turing (T4) has no native bf16 tensor cores; the code auto-selects fp16 on T4 and bf16 on Ampere+ by compute capability.

4. **`(1 − MFU)` overstates recoverable waste.** Because achievable MFU is ~35–50%, the headroom-to-peak figure is aspirational. The cost model reports recoverable separately and leads with it.

## 5. Results

> This repository ships the method and the code, not fabricated measurements. The tables and the headline chart (`assets/headline_divergence.png`) are populated by running the experiment on a GPU. Replace the placeholders below with your readings.

**5.1 The metric divergence (single workload).**

| Metric | Reading |
| --- | --- |
| `nvidia-smi` GPU-Util | _run to fill_ |
| DCGM `SM_ACTIVE` | _run to fill_ |
| DCGM `PIPE_TENSOR_ACTIVE` | _run to fill_ |
| MFU | _run to fill_ |

**5.2 Sensitivity.** Expected qualitative shape (to be confirmed by your run): MFU rises with batch size up to a memory limit; fp16/bf16 substantially exceeds fp32; too few DataLoader workers depress MFU while GPU-Util can remain high (the input-starvation signature).

**5.3 Dollars.** With measured MFU plugged into `analysis/dollar_model.py`, report total recoverable spend across an illustrative fleet.

## 6. Discussion: the product gap

Platforms in this space optimize and report the allocation layer. The work-efficiency layer is:

- **Measurable today** from the DCGM signal the GPU Operator already collects;
- **Invisible** in the current headline metric, which is allocation utilization;
- **Material**, because a job can be allocated and "100% utilized" while doing a fraction of useful work.

The proposed surface (mocked in `panel/app.py`) turns the existing signal into: per-workload MFU and goodput; a fleet-wide efficiency rollup; a dollar-waste leaderboard sorted by recoverable spend; and auto-flags with a likely cause (small batch, input starvation, precision left on the table). The recommendation is not "add a capability you lack" but "productize the signal you already collect, and stop leading with the metric that hides the larger loss."

## 7. Threats to validity

Single GPU and synthetic data make results directional, not a production benchmark. MFU depends on a correct FLOP model and a correct dense, precision-matched peak; both are addressed in §4 but should be re-verified per card. DCGM counters require root and the host engine. AMP mixes precisions; MFU is computed against the matmul peak. The framing assumes the allocation-side gains these platforms report are real and independent of work-efficiency, which is consistent with how such 30%→60% claims are typically described.

## 8. Related work

MFU: PaLM (Chowdhery et al., 2022); nanoGPT (Karpathy). "GPU utilization is misleading": Trainy; Modal; SM-efficiency analyses. Allocation-side GPU sharing and SLA-aware scheduling: NVIDIA Run:ai / KAI Scheduler, Gandiva, Tiresias, Gavel. This work is deliberately scoped to the work-efficiency layer beneath the allocation layer those systems optimize.

## 9. Conclusion

"GPU utilization," as commonly reported, is a mirage: it tracks occupancy, not useful work, and the two diverge in ways that are fixable through configuration. For an AI infrastructure platform, the opportunity is not a new capability but a new surface — turning already-collected DCGM signal into MFU, dollar-attributed waste, and actionable flags, and reframing the headline metric accordingly. The measurement is small and reproducible; the framing is, we argue, the valuable part.

---

## Reproducibility

All code, the verified peak table, and the cost model are in this repository. See [README.md](README.md) for one-command reproduction on a free Colab T4. The repository contains no fabricated measurements; every reported number is produced by running the provided scripts on the reader's own hardware.

## Author

Sanmati Sawalwade — MS Information Systems, Northeastern University (Silicon Valley)
sawalwade.s@northeastern.edu · [linkedin.com/in/sanmati-sawalwade](https://linkedin.com/in/sanmati-sawalwade) · [sanmati1997.github.io](https://sanmati1997.github.io)
