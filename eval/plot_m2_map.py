"""Top-down (world x-y) plot of the GT camera trajectory and the M2 landmarks.

Landmarks are triangulated by the front-end using GT camera poses (M2 setup).
Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_m2_map [--out pysteam_vo/results/m2_topdown.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.frontend.map import Map
from pysteam_vo.frontend.tracker import Tracker


def build_gt_map(seq: EurocSequence, cfg: dict, start: int, end: int) -> Map:
    tracker = Tracker(seq.calib, cfg["frontend"])
    for i in range(start, end):
        if not seq.gt_valid[i]:
            continue
        T_cw = invert_se3(seq.T_wc_gt[i])
        if tracker.need_keyframe(T_cw):
            tracker.add_keyframe(i, int(seq.image_ts_ns[i]), seq.load_image(i), T_cw)
    return tracker.map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "m2_topdown.png")
    args = ap.parse_args()

    cfg = load_config()
    m2, pc = cfg["m2_check"], cfg["plots"]
    seq = EurocSequence.from_config(cfg)
    start, end = m2["start_frame"], m2["end_frame"]
    the_map = build_gt_map(seq, cfg, start, end)
    a = the_map.to_arrays()
    lens = the_map.track_lengths()

    idx = np.arange(start, end)
    traj = seq.T_wc_gt[idx[seq.gt_valid[idx]], :3, 3]
    kf_c = invert_se3(a.T_cw)[:, :3, 3]
    long_tr = lens >= m2["min_track_len"]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(a.p_w[~long_tr, 0], a.p_w[~long_tr, 1], s=1, c="0.75", alpha=0.4, lw=0,
               label=f"landmarks, 2 views ({(~long_tr).sum()})")
    ax.scatter(a.p_w[long_tr, 0], a.p_w[long_tr, 1], s=1.5, c="tab:blue", alpha=0.6, lw=0,
               label=f"landmarks, >= {m2['min_track_len']} views ({long_tr.sum()})")
    ax.plot(traj[:, 0], traj[:, 1], c="tab:red", lw=2, zorder=4, label="GT camera trajectory")
    ax.scatter(kf_c[:, 0], kf_c[:, 1], s=6, c="k", marker="o", lw=0, zorder=3, label=f"keyframes ({len(kf_c)})")
    ax.scatter(*traj[0, :2], s=60, c="tab:green", zorder=5, label=f"start (frame {start})")

    # Keep far triangulation outliers from squashing the view.
    lo, hi = np.percentile(a.p_w[:, :2], [pc["clip_percentile"], 100 - pc["clip_percentile"]], axis=0)
    lo, hi = np.minimum(lo, traj[:, :2].min(0)), np.maximum(hi, traj[:, :2].max(0))
    pad = pc["pad_m"]
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_title(f"{seq.name} frames {start}-{end}: GT trajectory + landmarks triangulated from GT poses")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="best", fontsize=8)
    for h in leg.legend_handles[:2]:           # enlarge the tiny landmark dots in the legend only
        h.set_sizes([20])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=pc["dpi"], bbox_inches="tight")
    print(f"saved {args.out}  ({len(a.lm_ids)} landmarks, {len(a.kf_ids)} keyframes)")


if __name__ == "__main__":
    main()
