"""
train_loop.py
-------------
PHASE 1, the core finding. Runs ONE real training loop on a single GPU and
captures the two numbers that are rock-solid measurable anywhere (including a
free Colab T4):

    GPU-Util  (nvidia-smi)  -> "a kernel is running", says ~100%
    MFU       (computed)    -> useful compute / GPU peak, the truth

These two alone prove the thesis: a GPU that looks pinned at 100% is doing a
fraction of useful work. The middle rung (DCGM SM-active / tensor-active) lives
in dcgm_capture.py because it needs a host you control, not vanilla Colab.

Run on Colab (T4):
    !nvidia-smi -L
    !python measure/train_loop.py --steps 200 --batch-size 8 --seq-len 256

Outputs:
    assets/headline_metrics.json   (the numbers)
    assets/headline_divergence.png (the chart that is the pitch)
"""

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_peaks import detect_gpu_key, peak_flops, supported_precisions


# --------------------------------------------------------------------------- #
# A compact decoder-only transformer (nanoGPT-shaped). Small enough to train
# fast on a T4, real enough that the FLOP accounting is honest.
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    seq_len: int = 256
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = nn.MultiheadAttention(
            c.n_embd, c.n_head, dropout=c.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )
        mask = torch.triu(torch.ones(c.seq_len, c.seq_len), diagonal=1).bool()
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        h = self.ln1(x)
        T = x.size(1)
        a, _ = self.attn(h, h, h, attn_mask=self.attn_mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.seq_len, c.n_embd)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        # Weight tying (GPT-2 standard): the output head shares the token-embedding
        # matrix. This is REQUIRED for honest FLOP accounting. The `6 * N` convention
        # (with N = non-embedding params, subtracting only position embeddings)
        # assumes the one large embedding-shaped tensor doing matmul work is the tied
        # head. Untied, the token embedding (a pure lookup doing ~0 FLOPs) is counted
        # as matmul work and inflates MFU by ~1.6x for this config. Tying fixes it.
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Karpathy's MFU convention subtracts position embeddings. The token
            # embedding is tied to the head (see __init__), so it is counted once
            # and correctly attributed to the head matmul, not double-counted.
            n -= self.pos.weight.numel()
        return n


def flops_per_token(model: GPT, include_attention: bool = True) -> float:
    """
    FLOPs per token for fwd+bwd.
    Base: 6 * N (the standard PaLM/Karpathy approximation; N = non-embedding params).
    Optional attention term: 12 * n_layer * n_embd * seq_len, which matters at
    longer sequence lengths and otherwise slightly UNDER-counts FLOPs (making MFU
    look worse than it is). Including it keeps you honest.
    """
    c = model.c
    n = model.num_params(non_embedding=True)
    f = 6 * n
    if include_attention:
        f += 12 * c.n_layer * c.n_embd * c.seq_len
    return float(f)


# --------------------------------------------------------------------------- #
# GPU-Util sampler: polls nvidia-smi in a background thread.
# --------------------------------------------------------------------------- #
class GpuUtilSampler:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []  # (t, util_percent)
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> float | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return float(out.decode().strip().splitlines()[0])
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            u = self._poll_once()
            if u is not None:
                self._samples.append((time.time(), u))
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def mean_between(self, t0: float, t1: float) -> float | None:
        vals = [u for (t, u) in self._samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- #
# Precision selection: pick a precision the GPU can actually accelerate.
# --------------------------------------------------------------------------- #
def choose_precision(gpu_key: str | None, requested: str) -> str:
    cc_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if requested != "auto":
        return requested
    if gpu_key and "bf16" in supported_precisions(gpu_key) and cc_major >= 8:
        return "bf16"   # Ampere+ : bf16 is the clean default (no GradScaler needed)
    return "fp16"       # Turing (T4) and fallback : fp16 tensor cores


def autocast_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, None)


# --------------------------------------------------------------------------- #
# Main measurement.
# --------------------------------------------------------------------------- #
def measure(args) -> dict:
    assert torch.cuda.is_available(), "No CUDA GPU visible. Run this on a GPU runtime."
    device = "cuda"
    dev_name = torch.cuda.get_device_name()
    gpu_key = detect_gpu_key(dev_name)

    precision = choose_precision(gpu_key, args.precision)
    ac_dtype = autocast_dtype(precision)

    cfg = GPTConfig(seq_len=args.seq_len, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    N = model.num_params(non_embedding=True)
    fpt = flops_per_token(model, include_attention=not args.no_attention_flops)

    def batch():
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.seq_len), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.seq_len), device=device)
        return x, y

    def step():
        x, y = batch()
        opt.zero_grad(set_to_none=True)
        if ac_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=ac_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    # Warmup (compile/caches/clocks settle) - not timed.
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    sampler = GpuUtilSampler(gpu_index=args.gpu_index)
    sampler.start()
    t0 = time.time()
    last_loss = None
    for _ in range(args.steps):
        last_loss = step()
    torch.cuda.synchronize()
    t1 = time.time()
    sampler.stop()

    elapsed = t1 - t0
    tokens = args.batch_size * cfg.seq_len * args.steps
    tokens_per_sec = tokens / elapsed
    achieved_flops = fpt * tokens_per_sec

    gpu_util_mean = sampler.mean_between(t0, t1)
    mfu = None
    peak = None
    if gpu_key is not None and precision in supported_precisions(gpu_key):
        peak = peak_flops(gpu_key, precision)
        mfu = achieved_flops / peak

    result = {
        "device_name": dev_name,
        "gpu_key": gpu_key,
        "precision": precision,
        "config": asdict(cfg),
        "batch_size": args.batch_size,
        "steps_timed": args.steps,
        "elapsed_s": round(elapsed, 4),
        "params_non_embedding": N,
        "flops_per_token": fpt,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "achieved_tflops": round(achieved_flops / 1e12, 3),
        "peak_tflops": round(peak / 1e12, 1) if peak else None,
        "gpu_util_percent_mean": round(gpu_util_mean, 1) if gpu_util_mean is not None else None,
        "mfu_percent": round(mfu * 100, 1) if mfu is not None else None,
        "final_loss": round(float(last_loss.item()), 4) if last_loss is not None else None,
        "note": (
            "MFU skipped: GPU not in gpu_peaks table for this precision. "
            "Add its dense peak from the NVIDIA data sheet."
            if mfu is None else
            "MFU uses the dense tensor peak matching the training precision."
        ),
    }
    return result


def make_chart(result: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    util = result.get("gpu_util_percent_mean")
    mfu = result.get("mfu_percent")
    labels, values, colors = [], [], []
    if util is not None:
        labels.append("nvidia-smi\nGPU-Util");  values.append(util);  colors.append("#9aa0a6")
    if mfu is not None:
        labels.append("MFU\n(true work)");      values.append(mfu);   colors.append("#1a73e8")

    if not values:
        print("No metrics to chart.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("percent")
    ax.set_title("Same workload, two very different numbers", fontsize=12)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    sub = f"{result['device_name']}  |  {result['precision']}  |  batch {result['batch_size']}  |  seq {result['config']['seq_len']}"
    ax.text(0.5, -0.16, sub, transform=ax.transAxes, ha="center", fontsize=8, color="#5f6368")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description="Capture GPU-Util vs MFU for one workload.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--precision", choices=["auto", "fp32", "tf32", "fp16", "bf16"], default="auto")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--no-attention-flops", action="store_true",
                   help="Use bare 6*N FLOPs/token instead of 6*N + attention term.")
    p.add_argument("--out-dir", default="assets")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    result = measure(args)

    print(json.dumps(result, indent=2))
    json_path = os.path.join(args.out_dir, "headline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {json_path}")

    make_chart(result, os.path.join(args.out_dir, "headline_divergence.png"))


if __name__ == "__main__":
    main()
