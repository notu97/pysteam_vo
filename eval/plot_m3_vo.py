"""Run monocular VO (M3) and plot it top-down against GT after Sim(3) alignment.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_m3_vo [--start 1000 --end 1800] [--out ...png]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.frontend.tracker import MonoVO


def run_vo(seq: EurocSequence, cfg: dict, start: int, end: int) -> tuple[MonoVO, float]:
    vo = MonoVO(seq.calib, cfg["frontend"])
    t0 = time.perf_counter()
    for i in range(start, min(end, len(seq))):
        vo.process(i, int(seq.image_ts_ns[i]), seq.load_image(i))
        if vo.state == "lost":
            break
    return vo, time.perf_counter() - t0


def evaluate(vo: MonoVO, seq: EurocSequence, start: int, end: int) -> dict:
    """Sim(3) ATE on camera centers over tracked frames that have GT."""
    idx, T_cw = vo.tracked_poses()
    ok = seq.gt_valid[idx]
    c_est = invert_se3(T_cw[ok])[:, :3, 3]
    c_gt = seq.T_wc_gt[idx[ok], :3, 3]
    n_frames = int(np.sum(seq.gt_valid[start:end]))
    r = ate_sim3(c_est, c_gt)
    r.update(idx=idx[ok], c_gt=c_gt, tracked_fraction=ok.sum() / max(1, n_frames),
             first_tracked=int(idx[0]) if len(idx) else -1)
    return r


def main() -> None:
    cfg = load_config()
    m3 = cfg["m3_check"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=m3["start_frame"])
    ap.add_argument("--end", type=int, default=m3["end_frame"])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "m3_vo_topdown.png")
    args = ap.parse_args()

    seq = EurocSequence.from_config(cfg)
    vo, secs = run_vo(seq, cfg, args.start, args.end)
    r = evaluate(vo, seq, args.start, args.end)
    n_kf = len(vo.map.keyframes)
    print(f"[M3] {seq.name} frames {args.start}-{args.end}  state={vo.state}  first tracked={r['first_tracked']}")
    print(f"[M3] tracked {r['tracked_fraction']:.3f}  keyframes {n_kf}  landmarks {len(vo.map.landmarks)}  "
          f"runtime {secs:.1f} s ({1e3 * secs / max(1, args.end - args.start):.1f} ms/frame)")
    print(f"[M3] ATE (Sim3) rmse {r['rmse']:.3f} m  median {r['median']:.3f} m  max {r['max']:.3f} m  "
          f"scale {r['scale']:.3f}")

    pc = cfg["plots"]
    a = vo.map.to_arrays()
    lm = r["scale"] * a.p_w @ r["R"].T + r["t"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(lm[:, 0], lm[:, 1], s=1, c="tab:blue", alpha=0.3, lw=0, label=f"VO landmarks, aligned ({len(lm)})")
    ax.plot(r["c_gt"][:, 0], r["c_gt"][:, 1], c="k", lw=2, label="GT camera trajectory")
    ax.plot(r["aligned"][:, 0], r["aligned"][:, 1], c="tab:red", lw=1.5,
            label=f"VO (Sim3-aligned), ATE rmse {r['rmse']:.3f} m")
    ax.scatter(*r["c_gt"][0, :2], s=60, c="tab:green", zorder=5, label=f"start (frame {r['first_tracked']})")
    traj = np.vstack([r["c_gt"], r["aligned"]])[:, :2]
    lo, hi = np.percentile(lm[:, :2], [pc["clip_percentile"], 100 - pc["clip_percentile"]], axis=0)
    lo, hi = np.minimum(lo, traj.min(0)), np.maximum(hi, traj.max(0))
    ax.set_xlim(lo[0] - pc["pad_m"], hi[0] + pc["pad_m"])
    ax.set_ylim(lo[1] - pc["pad_m"], hi[1] + pc["pad_m"])
    ax.set_aspect("equal")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_title(f"{seq.name} frames {args.start}-{args.end}: monocular VO (init + PnP) vs GT")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="best", fontsize=8)
    leg.legend_handles[0].set_sizes([20])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=pc["dpi"], bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
