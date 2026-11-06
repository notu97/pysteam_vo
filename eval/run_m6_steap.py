"""M6: VO (M3) -> STEAP-style GP trajectory estimation in pysteam; compare with VO and BA (M5).

The STEAP problem is initialized from the VO keyframe map (poses, landmarks) and VO
finite-difference velocities, never from BA. BA is run separately, from the same VO map,
only as a comparison.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_m6_steap [--start 2600 --end 3400] [--no-ba] [--out ...png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.backend.problem import build_ba_problem, build_steap_problem
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.plot_m3_vo import run_vo


def _centers(T_cw: np.ndarray) -> np.ndarray:
    return invert_se3(T_cw)[:, :3, 3]


def run_m6(seq: EurocSequence, cfg: dict, start: int, end: int, with_ba: bool = True, verbose: bool = False) -> dict:
    vo, vo_secs = run_vo(seq, cfg, start, end)
    arr = vo.map.to_arrays()
    nonkf = vo.nonkeyframe_observations()

    # keyframe evaluation set
    kf_ok = seq.gt_valid[arr.kf_frame_idx]
    c_gt_kf = seq.T_wc_gt[arr.kf_frame_idx[kf_ok], :3, 3]
    out = {"vo": vo, "arr": arr, "vo_seconds": vo_secs, "c_gt_kf": c_gt_kf,
           "ate_kf_vo": ate_sim3(_centers(arr.T_cw[kf_ok]), c_gt_kf)}

    # all tracked frames inside the knot span: VO per-frame poses vs STEAP interpolation
    fidx, T_vo = vo.tracked_poses()
    t_f = seq.image_ts_ns[fidx]
    inside = (t_f >= arr.kf_t_ns[0]) & (t_f <= arr.kf_t_ns[-1]) & seq.gt_valid[fidx]
    fidx, T_vo, t_f = fidx[inside], T_vo[inside], t_f[inside]
    c_gt_all = seq.T_wc_gt[fidx, :3, 3]
    out.update(frame_idx=fidx, c_gt_all=c_gt_all, ate_all_vo=ate_sim3(_centers(T_vo), c_gt_all))

    # STEAP from the VO map
    sp = build_steap_problem(arr, nonkf, seq.calib.K, cfg["backend"])
    assert np.array_equal(sp.poses(), arr.T_cw), "STEAP must start from the VO keyframe poses"
    st = solve_lm(sp.problem, cfg["backend"], verbose=verbose, n_pose=sp.n_pose_dims)
    T_steap_all = sp.interpolate(t_f)
    out.update(steap=sp, stats_steap=st,
               ate_kf_steap=ate_sim3(_centers(sp.poses()[kf_ok]), c_gt_kf),
               ate_all_steap=ate_sim3(_centers(T_steap_all), c_gt_all))

    if with_ba:                                   # comparison only, also from the VO map
        ba = build_ba_problem(arr, seq.calib.K, cfg["backend"])
        st_ba = solve_lm(ba.problem, cfg["backend"], n_pose=ba.n_pose_dims)
        out.update(ba=ba, stats_ba=st_ba, ate_kf_ba=ate_sim3(_centers(ba.poses()[kf_ok]), c_gt_kf))
    return out


def report(r: dict, name: str, start: int, end: int) -> None:
    sp, st = r["steap"], r["stats_steap"]
    print(f"[M6] {name} frames {start}-{end}: knots {len(sp.pose_vars)}  landmarks {len(sp.lm_vars)}  "
          f"obs kf {sp.n_obs_kf} + non-kf {sp.n_obs_nonkf}  priors {sp.n_prior}")
    print(f"[M6] STEAP LM: cost {st.initial_cost:.0f} -> {st.final_cost:.0f}  {st.iterations} it "
          f"({st.termination})  {st.seconds:.0f} s")
    if "stats_ba" in r:
        print(f"[M6] BA    LM: {r['stats_ba'].iterations} it  {r['stats_ba'].seconds:.0f} s")
    ba = f"  BA {r['ate_kf_ba']['rmse']:.3f}" if "ate_kf_ba" in r else ""
    print(f"[M6] keyframe ATE (Sim3):   VO {r['ate_kf_vo']['rmse']:.3f}{ba}  STEAP {r['ate_kf_steap']['rmse']:.3f} m")
    print(f"[M6] all-frame ATE (Sim3):  VO {r['ate_all_vo']['rmse']:.3f}  STEAP(interp) {r['ate_all_steap']['rmse']:.3f} m "
          f"({len(r['frame_idx'])} frames)")


def main() -> None:
    cfg = load_config()
    m6 = cfg["m6_check"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=m6["start_frame"])
    ap.add_argument("--end", type=int, default=m6["end_frame"])
    ap.add_argument("--no-ba", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "m6_steap_topdown.png")
    args = ap.parse_args()

    seq = EurocSequence.from_config(cfg)
    r = run_m6(seq, cfg, args.start, args.end, with_ba=not args.no_ba, verbose=args.verbose)
    report(r, seq.name, args.start, args.end)

    pc = cfg["plots"]
    a = r["ate_all_steap"]
    lm = a["scale"] * r["steap"].points() @ a["R"].T + a["t"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(lm[:, 0], lm[:, 1], s=1, c="tab:blue", alpha=0.25, lw=0, label=f"STEAP landmarks, aligned ({len(lm)})")
    ax.plot(r["c_gt_all"][:, 0], r["c_gt_all"][:, 1], c="k", lw=2, label="GT (all frames)")
    ax.plot(r["ate_all_vo"]["aligned"][:, 0], r["ate_all_vo"]["aligned"][:, 1], c="tab:orange", lw=1.0,
            label=f"VO (all frames), ATE {r['ate_all_vo']['rmse']:.3f} m")
    ax.plot(a["aligned"][:, 0], a["aligned"][:, 1], c="tab:red", lw=1.5,
            label=f"STEAP (GP-interpolated), ATE {a['rmse']:.3f} m")
    traj = np.vstack([r["c_gt_all"], a["aligned"], r["ate_all_vo"]["aligned"]])[:, :2]
    lo, hi = np.percentile(lm[:, :2], [pc["clip_percentile"], 100 - pc["clip_percentile"]], axis=0)
    lo, hi = np.minimum(lo, traj.min(0)), np.maximum(hi, traj.max(0))
    ax.set_xlim(lo[0] - pc["pad_m"], hi[0] + pc["pad_m"])
    ax.set_ylim(lo[1] - pc["pad_m"], hi[1] + pc["pad_m"])
    ax.set_aspect("equal")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_title(f"{seq.name} frames {args.start}-{args.end}: STEAP (GP prior) vs VO vs GT")
    ax.grid(alpha=0.3)
    leg = ax.legend(loc="best", fontsize=8)
    leg.legend_handles[0].set_sizes([20])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=pc["dpi"], bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
