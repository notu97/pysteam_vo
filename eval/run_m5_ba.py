"""M5: run VO (M3), then keyframe bundle adjustment in pysteam; report and plot.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_m5_ba [--start 1000 --end 1800] [--out ...png]

ATE is computed on keyframe camera centers for both VO and BA (same frames, Sim(3)).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.backend.problem import build_ba_problem
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.plot_m3_vo import run_vo


def run_m5(seq: EurocSequence, cfg: dict, start: int, end: int, verbose: bool = False) -> dict:
    vo, vo_secs = run_vo(seq, cfg, start, end)
    arr = vo.map.to_arrays()
    ok = seq.gt_valid[arr.kf_frame_idx]
    c_gt = seq.T_wc_gt[arr.kf_frame_idx[ok], :3, 3]
    ate_vo = ate_sim3(invert_se3(arr.T_cw[ok])[:, :3, 3], c_gt)

    ba = build_ba_problem(arr, seq.calib.K, cfg["backend"])
    stats = solve_lm(ba.problem, cfg["backend"], verbose=verbose, n_pose=ba.n_pose_dims)
    T_ba = ba.poses()
    ate_ba = ate_sim3(invert_se3(T_ba[ok])[:, :3, 3], c_gt)
    return {"vo": vo, "arr": arr, "ba": ba, "stats": stats, "T_cw_ba": T_ba, "c_gt": c_gt,
            "ate_vo": ate_vo, "ate_ba": ate_ba, "n_kf": len(arr.kf_ids), "n_lm": len(ba.lm_vars),
            "n_obs": ba.n_obs, "vo_seconds": vo_secs}


def main() -> None:
    cfg = load_config()
    m5 = cfg["m5_check"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=m5["start_frame"])
    ap.add_argument("--end", type=int, default=m5["end_frame"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "m5_ba_topdown.png")
    args = ap.parse_args()

    seq = EurocSequence.from_config(cfg)
    r = run_m5(seq, cfg, args.start, args.end, args.verbose)
    st = r["stats"]
    print(f"[M5] {seq.name} frames {args.start}-{args.end}: KFs {r['n_kf']} landmarks {r['n_lm']} obs {r['n_obs']}")
    print(f"[M5] LM: cost {st.initial_cost:.1f} -> {st.final_cost:.1f}  {st.iterations} iterations "
          f"({st.termination})  {st.seconds:.1f} s  (VO {r['vo_seconds']:.1f} s)")
    print(f"[M5] keyframe ATE (Sim3) rmse: VO {r['ate_vo']['rmse']:.3f} m -> BA {r['ate_ba']['rmse']:.3f} m  "
          f"| median {r['ate_vo']['median']:.3f} -> {r['ate_ba']['median']:.3f}  "
          f"| max {r['ate_vo']['max']:.3f} -> {r['ate_ba']['max']:.3f}")

    pc = cfg["plots"]
    a = r["ate_ba"]
    lm = a["scale"] * r["ba"].points() @ a["R"].T + a["t"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(lm[:, 0], lm[:, 1], s=1, c="tab:blue", alpha=0.3, lw=0, label=f"BA landmarks, aligned ({len(lm)})")
    ax.plot(r["c_gt"][:, 0], r["c_gt"][:, 1], c="k", lw=2, label="GT (keyframes)")
    ax.plot(r["ate_vo"]["aligned"][:, 0], r["ate_vo"]["aligned"][:, 1], c="tab:orange", lw=1.2,
            label=f"VO, ATE {r['ate_vo']['rmse']:.3f} m")
    ax.plot(a["aligned"][:, 0], a["aligned"][:, 1], c="tab:red", lw=1.5, label=f"BA, ATE {a['rmse']:.3f} m")
    traj = np.vstack([r["c_gt"], a["aligned"], r["ate_vo"]["aligned"]])[:, :2]
    lo, hi = np.percentile(lm[:, :2], [pc["clip_percentile"], 100 - pc["clip_percentile"]], axis=0)
    lo, hi = np.minimum(lo, traj.min(0)), np.maximum(hi, traj.max(0))
    ax.set_xlim(lo[0] - pc["pad_m"], hi[0] + pc["pad_m"])
    ax.set_ylim(lo[1] - pc["pad_m"], hi[1] + pc["pad_m"])
    ax.set_aspect("equal")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_title(f"{seq.name} frames {args.start}-{args.end}: keyframe BA (pysteam) vs VO vs GT")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="best", fontsize=8)
    leg.legend_handles[0].set_sizes([20])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=pc["dpi"], bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
