"""
gpu_peaks.py
------------
The MFU denominator. This is the single most error-prone number in the whole
case study: if you divide achieved FLOPs by the wrong peak, your headline number
is off by up to ~8x and the pitch dies on contact.

Rule: divide by the peak that MATCHES the precision you trained in.
  - Trained in true fp32 (CUDA cores)        -> use the fp32 peak
  - Trained in fp16/bf16 with tensor cores   -> use the fp16/bf16 tensor peak
  - Trained in tf32 (Ampere+ default matmul) -> use the tf32 tensor peak

All numbers below are DENSE tensor-core peaks (no 2:4 sparsity), in TFLOPS.
Sources: NVIDIA data sheets / product briefs (T4, L4, A100 verified Jun 2026).
Confirm the exact figure for whatever card you actually rent before publishing.
"""

# peak[gpu_key][precision] = TFLOPS (dense, no sparsity)
PEAKS_TFLOPS = {
    # Turing. NOTE: T4 has NO native bf16 tensor cores. Use fp16 here, not bf16.
    "T4": {
        "fp32": 8.1,
        "fp16": 65.0,   # tensor core
        # tf32 / bf16 not natively accelerated on Turing -> intentionally absent
    },
    # Ada. Has tf32, fp16, bf16 tensor cores.
    "L4": {
        "fp32": 30.3,
        "tf32": 60.0,   # tensor core
        "fp16": 121.0,  # tensor core (dense)
        "bf16": 121.0,  # tensor core (dense)
    },
    # Ampere. Commonly cited dense figures; confirm against your rental's sheet.
    "A10": {
        "fp32": 31.2,
        "tf32": 62.5,
        "fp16": 125.0,
        "bf16": 125.0,
    },
    # Ampere flagship, for scale-up reference.
    "A100": {
        "fp32": 19.5,
        "tf32": 156.0,
        "fp16": 312.0,
        "bf16": 312.0,
    },
}

# Map common device-name substrings to a key above.
_NAME_HINTS = [
    ("a100", "A100"),
    ("a10", "A10"),
    ("l4", "L4"),
    ("t4", "T4"),
]


def detect_gpu_key(device_name: str) -> str | None:
    """Best-effort match from torch.cuda.get_device_name() to a table key."""
    n = (device_name or "").lower()
    for hint, key in _NAME_HINTS:
        if hint in n:
            return key
    return None


def peak_flops(gpu_key: str, precision: str) -> float:
    """Return peak FLOPs (not TFLOPs) for a (gpu, precision) pair."""
    table = PEAKS_TFLOPS.get(gpu_key)
    if table is None:
        raise KeyError(
            f"Unknown GPU '{gpu_key}'. Add its dense peaks to PEAKS_TFLOPS "
            f"from the NVIDIA data sheet, then re-run."
        )
    if precision not in table:
        raise KeyError(
            f"GPU '{gpu_key}' has no listed peak for precision '{precision}'. "
            f"Available: {sorted(table)}. (Turing T4 has no native bf16/tf32.)"
        )
    return table[precision] * 1e12


def supported_precisions(gpu_key: str) -> list[str]:
    return sorted(PEAKS_TFLOPS.get(gpu_key, {}).keys())
