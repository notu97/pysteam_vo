"""M7: do GP-interpolated poses match solved poses on held-out keyframes?

1. Full solve: STEAP (M6) with every VO keyframe as a knot -> reference keyframe poses.
2. Hold-out solve: every N-th keyframe (never the first or last) is removed as a knot; its
   observations are kept, attached through the GP-interpolated pose like any other
   in-between frame. Same landmarks and scale anchor as the full solve, so both solutions
   share one gauge and are compared directly, without alignment.
3. At each held-out keyframe time: GP-interpolated pose (hold-out solve) vs solved pose
   (full solve), plus a naive baseline (SE(3) linear interpolation between the same
   hold-out knots), plus both against GT (via the full solve's Sim(3) alignment).

Both solves start from the VO map (M3), as in M6.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_m7_holdout [--start 2600 --end 3400] [--every 3] [--out ...png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from pysteam_vo.backend.problem import build_steap_problem
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, interpolate_poses, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.plot_m3_vo import run_vo
from pysteam_vo.frontend.map import MapArrays


def holdout_split(arr: MapArrays, nonkf: tuple, every: int) -> tuple[MapArrays, tuple, np.ndarray]:
    """Remove every `every`-th keyframe (not first/last) as a knot; move its observations to nonkf.

    Landmark arrays are unchanged, so landmark rows stay valid. Returns
    (arr_holdout, nonkf_holdout, held_out keyframe indices into `arr`).
    """
    n = len(arr.kf_ids)
    held = np.arange(every, n - 1, every)
    keep = np.setdiff1d(np.arange(n), held)
    new_idx = -np.ones(n, dtype=np.int64)
    new_idx[keep] = np.arange(len(keep))
    obs_keep = np.isin(arr.obs_kf, keep)
    arr_ho = MapArrays(arr.kf_ids[keep], arr.kf_frame_idx[keep], arr.kf_t_ns[keep], arr.T_cw[keep],
                       arr.lm_ids, arr.p_w, new_idx[arr.obs_kf[obs_keep]], arr.obs_lm[obs_keep],
                       arr.obs_uv[obs_keep])
    moved = ~obs_keep
    t, lm, uv = nonkf
    nonkf_ho = (np.concatenate([t, arr.kf_t_ns[arr.obs_kf[moved]]]),
                np.concatenate([lm, arr.lm_ids[arr.obs_lm[moved]]]),
                np.concatenate([uv, arr.obs_uv[moved]]))
    return arr_ho, nonkf_ho, held


def _rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Per-pose angle of Ra^T Rb, (N,) degrees."""
    return np.degrees(np.linalg.norm(Rotation.from_matrix(np.swapaxes(Ra, -1, -2) @ Rb).as_rotvec(), axis=-1))


def run_m7(seq: EurocSequence, cfg: dict, start: int, end: int, every: int) -> dict:
    vo, _ = run_vo(seq, cfg, start, end)
    arr = vo.map.to_arrays()
    nonkf = vo.nonkeyframe_observations()
    be = cfg["backend"]

    full = build_steap_problem(arr, nonkf, seq.calib.K, be)
    assert np.array_equal(full.poses(), arr.T_cw), "M7 must start from the VO map"
    st_full = solve_lm(full.problem, be, n_pose=full.n_pose_dims)

    arr_ho, nonkf_ho, held = holdout_split(arr, nonkf, every)
    ho = build_steap_problem(arr_ho, nonkf_ho, seq.calib.K, be,
                             landmark_rows=full.lm_rows, anchor_row=full.anchor_row)
    assert np.array_equal(ho.lm_rows, full.lm_rows) and ho.anchor_row == full.anchor_row
    st_ho = solve_lm(ho.problem, be, n_pose=ho.n_pose_dims)

    t_held = arr.kf_t_ns[held]
    T_ref = full.poses()[held]                                 # solved poses (full solve)
    T_gp = ho.interpolate(t_held)                              # GP interpolation (hold-out solve)
    T_lin_wc, ok = interpolate_poses(arr_ho.kf_t_ns, invert_se3(ho.poses()), t_held, max_gap_ns=2**62)
    assert ok.all()
    T_lin = invert_se3(T_lin_wc)                               # naive SE(3) interpolation, same knots

    # metres: Sim(3) of the full solve's keyframes to GT
    kf_ok = seq.gt_valid[arr.kf_frame_idx]
    al = ate_sim3(invert_se3(full.poses()[kf_ok])[:, :3, 3], seq.T_wc_gt[arr.kf_frame_idx[kf_ok], :3, 3])
    s, R_al, t_al = al["scale"], al["R"], al["t"]
    c = lambda T: invert_se3(T)[:, :3, 3]

    def vs(T_a, T_b):
        return s * np.linalg.norm(c(T_a) - c(T_b), axis=1), _rot_err_deg(T_a[:, :3, :3], T_b[:, :3, :3])

    # Local (drift-free) comparison: each held-out pose relative to the preceding retained
    # knot, in each solve. Cancels any slow divergence between the two solutions so only the
    # interpolation itself is measured.
    prev = held - 1                                            # retained (holdout never takes neighbours)
    assert not np.isin(prev, held).any()
    T_prev_full = full.poses()[prev]
    ho_idx = {int(f): i for i, f in enumerate(arr_ho.kf_ids)}
    T_prev_ho = ho.poses()[[ho_idx[int(f)] for f in arr.kf_ids[prev]]]

    def local(T_held, T_prev):
        return T_held @ invert_se3(T_prev)                     # T_{h,prev}

    def vs_local(T_est):
        A, B = local(T_est, T_prev_ho), local(T_ref, T_prev_full)
        return s * np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1), _rot_err_deg(A[:, :3, :3], B[:, :3, :3])

    gt_ok = seq.gt_valid[arr.kf_frame_idx[held]]
    T_gt_wc = seq.T_wc_gt[arr.kf_frame_idx[held][gt_ok]]

    def vs_gt(T_cw):
        T_wc = invert_se3(T_cw[gt_ok])
        p = s * T_wc[:, :3, 3] @ R_al.T + t_al
        R = R_al @ T_wc[:, :3, :3]
        return np.linalg.norm(p - T_gt_wc[:, :3, 3], axis=1), _rot_err_deg(R, T_gt_wc[:, :3, :3])

    return {
        "n_kf": len(arr.kf_ids), "n_held": len(held), "held": held, "t_held": t_held,
        "every": every, "stats_full": st_full, "stats_ho": st_ho,
        "gp_vs_solved": vs(T_gp, T_ref), "lin_vs_solved": vs(T_lin, T_ref),
        "gp_vs_solved_local": vs_local(T_gp), "lin_vs_solved_local": vs_local(T_lin),
        "solved_vs_gt": vs_gt(T_ref), "gp_vs_gt": vs_gt(T_gp), "lin_vs_gt": vs_gt(T_lin),
        "kf_dt_s": np.diff(arr_ho.kf_t_ns) * 1e-9,
    }


def _fmt(err: tuple[np.ndarray, np.ndarray]) -> str:
    dt, dr = err
    return (f"trans median {1e3 * np.median(dt):6.1f} mm  p90 {1e3 * np.percentile(dt, 90):6.1f} mm | "
            f"rot median {np.median(dr):.3f} deg  p90 {np.percentile(dr, 90):.3f} deg")


def report(r: dict, name: str, start: int, end: int) -> None:
    print(f"[M7] {name} frames {start}-{end}: {r['n_kf']} keyframes, {r['n_held']} held out (every {r['every']}); "
          f"hold-out knot spacing median {np.median(r['kf_dt_s']):.2f} s")
    print(f"[M7] solves: full {r['stats_full'].iterations} it {r['stats_full'].seconds:.0f} s, "
          f"hold-out {r['stats_ho'].iterations} it {r['stats_ho'].seconds:.0f} s")
    print(f"[M7] LOCAL (relative to preceding knot, drift-free):")
    print(f"[M7]   GP interp     vs solved: {_fmt(r['gp_vs_solved_local'])}")
    print(f"[M7]   linear interp vs solved: {_fmt(r['lin_vs_solved_local'])}")
    print(f"[M7] GLOBAL (includes slow divergence between the two solves):")
    print(f"[M7]   GP interp     vs solved: {_fmt(r['gp_vs_solved'])}")
    print(f"[M7]   linear interp vs solved: {_fmt(r['lin_vs_solved'])}")
    print(f"[M7] vs GT (rotation column is dominated by the position-fitted alignment; see notes):")
    print(f"[M7] solved        vs GT:     {_fmt(r['solved_vs_gt'])}")
    print(f"[M7] GP interp     vs GT:     {_fmt(r['gp_vs_gt'])}")
    print(f"[M7] linear interp vs GT:     {_fmt(r['lin_vs_gt'])}")


def main() -> None:
    cfg = load_config()
    m7 = cfg["m7_check"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=m7["start_frame"])
    ap.add_argument("--end", type=int, default=m7["end_frame"])
    ap.add_argument("--every", type=int, default=m7["holdout_every"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(__file__).resolve().parents[1] / "results" / f"m7_holdout_{args.start}_{args.end}.png"

    seq = EurocSequence.from_config(cfg)
    r = run_m7(seq, cfg, args.start, args.end, args.every)
    report(r, seq.name, args.start, args.end)

    t = (r["t_held"] - r["t_held"][0]) * 1e-9
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for (key, lab, col) in [("gp_vs_solved", "GP interpolation", "tab:red"),
                            ("lin_vs_solved", "linear SE(3) interpolation", "tab:gray")]:
        axs[0].plot(t, 1e3 * r[key + "_local"][0], ".-", c=col, lw=1.0, ms=3, label=f"{lab} (local)")
        axs[1].plot(t, r[key + "_local"][1], ".-", c=col, lw=1.0, ms=3, label=f"{lab} (local)")
        axs[0].plot(t, 1e3 * r[key][0], ":", c=col, lw=0.8, label=f"{lab} (global)")
        axs[1].plot(t, r[key][1], ":", c=col, lw=0.8, label=f"{lab} (global)")
    axs[0].set_ylabel("translation error [mm]")
    axs[1].set_ylabel("rotation error [deg]")
    axs[1].set_xlabel("time since first held-out keyframe [s]")
    axs[0].set_title(f"{seq.name} frames {args.start}-{args.end}: interpolated vs solved pose at "
                     f"{r['n_held']} held-out keyframes")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
