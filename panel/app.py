"""
panel/app.py
------------
PHASE 4, the "what to build". A mockup of the management-plane surface most GPU
platforms are missing: per-workload MFU and goodput (not just allocation %), a
fleet-wide efficiency rollup, a dollar-waste leaderboard, and auto-flags with cause.

You do NOT ship this. You show what their panel could be.

This will NOT run inline in Colab. Run it locally or deploy to Vercel/Streamlit:
    pip install streamlit
    streamlit run panel/app.py

It reads measured numbers from ../assets/*.json when present, and otherwise
falls back to an illustrative fleet so the mockup always renders.
"""

import json
import os
import sys

import streamlit as st

# Make repo-root modules importable when run as `streamlit run panel/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.dollar_model import Job, fleet_report  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def load_measured_mfu() -> float | None:
    path = os.path.join(ASSETS, "headline_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        mfu = d.get("mfu_percent")
        return mfu / 100 if mfu is not None else None
    return None


def likely_cause(mfu_pct, gpu_util_pct, batch_size):
    if gpu_util_pct is not None and gpu_util_pct < 70:
        return "input starvation (dataloader stall)"
    if batch_size is not None and batch_size <= 4:
        return "small batch (low arithmetic intensity)"
    if mfu_pct is not None and mfu_pct < 25:
        return "fp32 where bf16 would do, or short sequences"
    return "review config"


def main():
    st.set_page_config(page_title="GPU work-efficiency surface (mockup)", layout="wide")
    st.title("Work-efficiency surface")
    st.caption("Proposed work-efficiency panel mockup. Allocation utilization tells you a GPU is "
               "busy. This tells you whether it is doing useful work.")

    measured = load_measured_mfu()
    target = st.sidebar.slider("Achievable MFU target", 0.20, 0.60, 0.45, 0.01)
    cost = st.sidebar.number_input("GPU $/hour", value=2.0, step=0.5)

    # Illustrative fleet. The first job uses the measured MFU if available.
    m0 = measured if measured is not None else 0.22
    jobs = [
        Job("team-vision/finetune", gpu_hours=100 * 24 * 30, mfu=m0, gpu_hourly_cost=cost),
        Job("team-nlp/pretrain",    gpu_hours=40 * 24 * 30,  mfu=0.41, gpu_hourly_cost=cost),
        Job("team-rec/embeddings",  gpu_hours=20 * 24 * 30,  mfu=0.08, gpu_hourly_cost=cost),
    ]
    report = fleet_report(jobs, mfu_target=target)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fleet spend / mo", f"${report['total_spend_$']:,.0f}")
    c2.metric("Recoverable / mo", f"${report['total_recoverable_$']:,.0f}",
              help="Same work retuned to the achievable target. The defensible number.")
    avg = sum(j.mfu for j in jobs) / len(jobs) * 100
    c3.metric("Avg MFU", f"{avg:.0f}%",
              delta=f"{'measured' if measured is not None else 'illustrative'}")

    st.subheader("Dollar-waste leaderboard")
    st.caption("Which jobs leak the most reclaimable spend. Sorted by recoverable $.")
    st.dataframe(report["leaderboard"], use_container_width=True)

    st.subheader("Auto-flags")
    bs = None
    util = None
    hp = os.path.join(ASSETS, "headline_metrics.json")
    if os.path.exists(hp):
        with open(hp) as f:
            d = json.load(f)
        bs = d.get("batch_size")
        util = d.get("gpu_util_percent_mean")
    for row in report["leaderboard"]:
        if row["mfu_percent"] < target * 100 * 0.8:
            cause = likely_cause(row["mfu_percent"], util, bs)
            st.warning(
                f"**{row['job']}** - MFU {row['mfu_percent']:.0f}%, likely cause: {cause}. "
                f"Est. ${row['recoverable_$']:,.0f}/mo reclaimable."
            )

    st.divider()
    st.caption("Honest note: MFU is a known metric (PaLM/Karpathy, Trainy). The gap is "
               "productization. DCGM already ships in the NVIDIA GPU Operator these "
               "platforms build on; the signal exists, the surface does not.")


if __name__ == "__main__":
    main()
