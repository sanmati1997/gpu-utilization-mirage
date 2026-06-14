"""
sweep.py
--------
PHASE 2, the "so what". Hold the model fixed, vary ONE knob, watch MFU move.
This proves the waste is fixable by configuration, not inherent to the hardware,
which is what makes surfacing it worth a product surface.

Modes:
    --mode batch        sweep batch size (the #1 MFU culprit)
    --mode precision    sweep precisions the GPU actually accelerates
    --mode dataloader   sweep DataLoader workers (shows GPU starvation on input)

Run on Colab (T4):
    !python measure/sweep.py --mode batch
    !python measure/sweep.py --mode precision

Outputs per mode:
    assets/sweep_<mode>.json
    assets/sweep_<mode>.png
"""

import argparse
import json
import os
import time

import torch

from gpu_peaks import detect_gpu_key, peak_flops, supported_precisions
from train_loop import (
    GPTConfig, GPT, flops_per_token, GpuUtilSampler,
    choose_precision, autocast_dtype,
)


def _synthetic_loader(batch_size, seq_len, vocab, num_workers, n_batches):
    """A CPU-bound dataset so that too-few workers visibly starve the GPU."""
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    class CpuBoundDS(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, _):
            # Deliberate CPU work to mimic real preprocessing/tokenization.
            a = np.random.rand(256, 256).astype("float32")
            _ = a @ a
            x = torch.randint(0, vocab, (seq_len,))
            y = torch.randint(0, vocab, (seq_len,))
            return x, y

    ds = CpuBoundDS(batch_size * n_batches)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))


def run_one(cfg: GPTConfig, batch_size: int, precision: str,
            steps: int, warmup: int, gpu_key: str | None,
            use_dataloader: bool = False, num_workers: int = 0) -> dict:
    device = "cuda"
    model = GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    ac_dtype = autocast_dtype(precision)
    fpt = flops_per_token(model, include_attention=True)

    def fwd_bwd(x, y):
        opt.zero_grad(set_to_none=True)
        if ac_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=ac_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if use_dataloader:
        loader = _synthetic_loader(batch_size, cfg.seq_len, cfg.vocab_size,
                                   num_workers, warmup + steps)
        it = iter(loader)

        def next_batch():
            x, y = next(it)
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        def next_batch():
            x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), device=device)
            y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), device=device)
            return x, y

    for _ in range(warmup):
        fwd_bwd(*next_batch())
    torch.cuda.synchronize()

    sampler = GpuUtilSampler()
    sampler.start()
    t0 = time.time()
    for _ in range(steps):
        fwd_bwd(*next_batch())
    torch.cuda.synchronize()
    t1 = time.time()
    sampler.stop()

    tokens = batch_size * cfg.seq_len * steps
    tps = tokens / (t1 - t0)
    achieved = fpt * tps
    util = sampler.mean_between(t0, t1)
    mfu = None
    if gpu_key and precision in supported_precisions(gpu_key):
        mfu = achieved / peak_flops(gpu_key, precision)

    del model, opt
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "precision": precision,
        "num_workers": num_workers if use_dataloader else None,
        "tokens_per_sec": round(tps, 1),
        "gpu_util_percent": round(util, 1) if util is not None else None,
        "mfu_percent": round(mfu * 100, 1) if mfu is not None else None,
    }


def make_chart(rows, x_key, x_label, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("mfu_percent") is not None]
    if not rows:
        print("No MFU rows to chart (GPU not in peaks table?).")
        return
    xs = [str(r[x_key]) for r in rows]
    mfu = [r["mfu_percent"] for r in rows]
    util = [r["gpu_util_percent"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, util, marker="o", color="#9aa0a6", label="GPU-Util")
    ax.plot(xs, mfu, marker="o", color="#1a73e8", label="MFU")
    ax.set_xlabel(x_label)
    ax.set_ylabel("percent")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=12)
    ax.legend()
    for x, v in zip(xs, mfu):
        ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=9, color="#1a73e8")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description="MFU sensitivity sweeps.")
    p.add_argument("--mode", choices=["batch", "precision", "dataloader"], default="batch")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--out-dir", default="assets")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Run on a GPU runtime."
    gpu_key = detect_gpu_key(torch.cuda.get_device_name())
    cfg = GPTConfig(seq_len=args.seq_len)
    rows = []

    if args.mode == "batch":
        prec = choose_precision(gpu_key, "auto")
        for bs in [1, 2, 4, 8, 16, 32]:
            try:
                rows.append(run_one(cfg, bs, prec, args.steps, args.warmup, gpu_key))
            except RuntimeError as e:  # OOM at large batch
                print(f"batch {bs}: {e}")
                break
        make_chart(rows, "batch_size", "batch size",
                   f"MFU vs batch size  ({prec}, {gpu_key or '?'})",
                   os.path.join(args.out_dir, "sweep_batch.png"))

    elif args.mode == "precision":
        precisions = supported_precisions(gpu_key) if gpu_key else ["fp16"]
        for prec in precisions:
            rows.append(run_one(cfg, 8, prec, args.steps, args.warmup, gpu_key))
        make_chart(rows, "precision", "precision",
                   f"MFU vs precision  (batch 8, {gpu_key or '?'})",
                   os.path.join(args.out_dir, "sweep_precision.png"))

    elif args.mode == "dataloader":
        prec = choose_precision(gpu_key, "auto")
        for nw in [0, 1, 2, 4, 8]:
            rows.append(run_one(cfg, 8, prec, args.steps, args.warmup, gpu_key,
                                use_dataloader=True, num_workers=nw))
        make_chart(rows, "num_workers", "dataloader workers",
                   f"MFU vs dataloader workers  ({prec}, {gpu_key or '?'})",
                   os.path.join(args.out_dir, "sweep_dataloader.png"))

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"sweep_{args.mode}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
