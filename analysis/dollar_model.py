"""
dollar_model.py
---------------
PHASE 3, metric -> business case. This is the screenshot line for a PM.

CORRECTION vs the naive model. The blueprint used:
    wasted_$ = gpu_hours * cost * (1 - MFU)
That overstates RECOVERABLE waste, because no real training job hits 100% MFU.
A well-tuned run typically lands ~30-50% MFU (PaLM reported ~46%). So (1 - MFU)
is "gap to the theoretical ceiling", not money you can actually get back.

We report BOTH, clearly labeled:
  headroom_vs_peak : gpu_hours * cost * (1 - mfu)               (aspirational ceiling)
  recoverable      : gpu_hours * cost * (1 - mfu / mfu_target)  (defensible ask)

The recoverable figure assumes the same work, retuned to mfu_target, finishes in
(mfu / mfu_target) of the time, so (1 - mfu/mfu_target) of the spend is reclaimable.
Lead the pitch with `recoverable`. Keep `headroom_vs_peak` only as context.
"""

from dataclasses import dataclass


@dataclass
class Job:
    name: str
    gpu_hours: float
    mfu: float                 # 0..1, measured
    gpu_hourly_cost: float = 2.0


def waste(job: Job, mfu_target: float = 0.45) -> dict:
    spend = job.gpu_hours * job.gpu_hourly_cost
    headroom = spend * (1.0 - job.mfu)
    if mfu_target <= 0:
        recoverable = 0.0
    else:
        recoverable = spend * max(0.0, 1.0 - job.mfu / mfu_target)
    return {
        "job": job.name,
        "spend_$": round(spend, 2),
        "mfu_percent": round(job.mfu * 100, 1),
        "headroom_vs_peak_$": round(headroom, 2),
        "recoverable_$": round(recoverable, 2),
    }


def fleet_report(jobs: list[Job], mfu_target: float = 0.45) -> dict:
    rows = [waste(j, mfu_target) for j in jobs]
    rows.sort(key=lambda r: r["recoverable_$"], reverse=True)  # leaderboard
    return {
        "mfu_target_percent": round(mfu_target * 100, 1),
        "total_spend_$": round(sum(r["spend_$"] for r in rows), 2),
        "total_recoverable_$": round(sum(r["recoverable_$"] for r in rows), 2),
        "leaderboard": rows,
    }


def _example():
    """Illustrative fleet. Replace mfu values with your measured numbers."""
    jobs = [
        Job("team-vision/finetune", gpu_hours=100 * 24 * 30, mfu=0.22),  # 100 GPUs, 1 month
        Job("team-nlp/pretrain",    gpu_hours=40 * 24 * 30,  mfu=0.41),
        Job("team-rec/embeddings",  gpu_hours=20 * 24 * 30,  mfu=0.08),
    ]
    return fleet_report(jobs, mfu_target=0.45)


if __name__ == "__main__":
    import json
    print(json.dumps(_example(), indent=2))
