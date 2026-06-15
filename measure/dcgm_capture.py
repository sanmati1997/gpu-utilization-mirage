"""
dcgm_capture.py
---------------
The MIDDLE RUNG of the ladder, captured with DCGM — the exact tool most GPU
platforms already ship (it is bundled in the NVIDIA GPU Operator they build on).
This is the "the signal is already in your stack" proof.

IMPORTANT: run this on a host where you control the OS (a rented A10/L4/A100 VM
where you are root, or your own cluster node). It will NOT work on a vanilla
Colab runtime: Colab does not let you run the DCGM host engine or read profiling
counters reliably. That limitation is exactly why the headline (train_loop.py,
GPU-Util vs MFU) is intentionally DCGM-free.

DCGM profiling fields sampled here:
    1001 DCGM_FI_PROF_GR_ENGINE_ACTIVE  -> compute engine active (the DCGM analog of GPU-Util)
    1002 DCGM_FI_PROF_SM_ACTIVE         -> fraction of time SMs have >=1 warp resident
    1004 DCGM_FI_PROF_PIPE_TENSOR_ACTIVE-> fraction of cycles the tensor pipe is active
                                           (the SHARPEST "is it doing matmul work" signal)

Usage (two terminals on the same VM):
    # terminal A: start a sustained workload
    python measure/train_loop.py --steps 100000 --batch-size 8 --seq-len 256
    # terminal B: sample DCGM for ~20s while A runs
    python measure/dcgm_capture.py --seconds 20 --gpu-index 0
"""

import argparse
import json
import os
import shutil
import subprocess

FIELDS = {
    "1001": "gr_engine_active",
    "1002": "sm_active",
    "1004": "tensor_active",
}


def dcgm_available() -> bool:
    return shutil.which("dcgmi") is not None


def run_dmon(seconds: float, gpu_index: int, delay_ms: int = 100) -> list[dict]:
    """Sample dcgmi dmon and return a list of per-sample dicts (values are 0..1)."""
    count = max(1, int((seconds * 1000) / delay_ms))
    field_ids = ",".join(FIELDS.keys())
    cmd = ["dcgmi", "dmon", "-i", str(gpu_index), "-e", field_ids,
           "-c", str(count), "-d", str(delay_ms)]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=seconds + 30)
    text = out.decode(errors="replace")

    # dcgmi dmon prints a header, then rows like:  GPU 0   0.998   0.551   0.213
    samples = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) >= 2 + len(FIELDS) and toks[0].upper() == "GPU":
            try:
                if int(toks[1]) != gpu_index:
                    continue
                nums = [float(t) for t in toks[2:2 + len(FIELDS)]]
            except ValueError:
                continue  # header or non-numeric row
            samples.append(dict(zip(FIELDS.values(), nums)))
    return samples


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {}
    out = {}
    for key in FIELDS.values():
        vals = [s[key] for s in samples if key in s]
        out[f"{key}_mean_percent"] = round(100 * sum(vals) / len(vals), 1) if vals else None
    out["n_samples"] = len(samples)
    return out


def main():
    p = argparse.ArgumentParser(description="Sample DCGM SM/tensor activity during a workload.")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--delay-ms", type=int, default=100)
    p.add_argument("--out-dir", default="assets")
    args = p.parse_args()

    if not dcgm_available():
        print(
            "dcgmi not found.\n"
            "This is expected on Colab. On a VM you control, install DCGM:\n"
            "  Ubuntu: sudo apt-get install -y datacenter-gpu-manager\n"
            "          sudo systemctl start nvidia-dcgm   (or: nv-hostengine -t off; nv-hostengine)\n"
            "Then re-run. Meanwhile the headline (GPU-Util vs MFU) already stands on its own."
        )
        return

    try:
        samples = run_dmon(args.seconds, args.gpu_index, args.delay_ms)
    except subprocess.CalledProcessError as e:
        print("dcgmi failed. Common cause: profiling counters need the host engine "
              "running and may require root. Output:\n" + e.output.decode(errors="replace"))
        return

    summary = summarize(samples)
    print(json.dumps(summary, indent=2))
    if summary:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "dcgm_metrics.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {path}")
        print(
            "\nLadder reading: gr_engine_active ~ GPU-Util (looks busy), "
            "sm_active a bit lower, tensor_active is the real tell. "
            "Pair tensor_active with MFU from train_loop.py for the full 3-rung chart."
        )


if __name__ == "__main__":
    main()
