"""S6: stereo STEAP vs S5 stereo BA on the SAME VO map; keyframe + all-frame ATE; plot vs GT.

Both start from the map written by eval/plot_s3_map.py (one S3 VO run, with non-keyframe
observations). STEAP never starts from BA.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_s6_stereo_steap --qcd-sweep      # tune qcd on s6_check.qcd_segment
    python -m pysteam_vo.eval.run_s6_stereo_steap [--qcd 1.0]       # full sequence: VO vs BA vs STEAP
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.backend.stereo_problem import build_stereo_ba_problem, build_stereo_steap_problem
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.export_tum import write_tum
from pysteam_vo.eval.run_s2_stereo_vo import ate
from pysteam_vo.eval.run_s5_stereo_ba import load_map
from pysteam_vo.frontend.map import MapArrays


def slice_map(arr: MapArrays, nonkf: tuple, frame_lo: int, frame_hi: int) -> tuple[MapArrays, tuple]:
    """Keyframes with frame index in [lo, hi) and their observations; landmark arrays unchanged."""
    keep = np.flatnonzero((arr.kf_frame_idx >= frame_lo) & (arr.kf_frame_idx < frame_hi))
    new = -np.ones(len(arr.kf_ids), np.int64)
    new[keep] = np.arange(len(keep))
    o = np.isin(arr.obs_kf, keep)
    sub = MapArrays(arr.kf_ids[keep], arr.kf_frame_idx[keep], arr.kf_t_ns[keep], arr.T_cw[keep], arr.lm_ids, arr.p_w,
                    new[arr.obs_kf[o]], arr.obs_lm[o], arr.obs_uv[o], arr.obs_uR[o])
    t, lm, uv = nonkf
    m = (t > sub.kf_t_ns[0]) & (t < sub.kf_t_ns[-1])
    return sub, (t[m], lm[m], uv[m])


def kf_ate(T_rw: np.ndarray, kf_f: np.ndarray, seq: StereoSequence) -> dict:
    ok = seq.gt_valid[kf_f]
    c = invert_se3(T_rw[ok])[:, :3, 3]
    G = seq.T_w_rect_gt[kf_f[ok], :3, 3]
    return {"se3": ate(c, G, False), "sim3": ate(c, G, True), "f": kf_f[ok]}


def run_steap(arr, nonkf, seq, cfg):
    sp = build_stereo_steap_problem(arr, nonkf, seq.rig.K, seq.rig.baseline, cfg["backend"])
    assert np.array_equal(sp.poses(), arr.T_cw), "STEAP must start from the VO keyframe poses"
    st = solve_lm(sp.problem, cfg["backend"], n_pose=sp.n_pose_dims)
    return sp, st


def qcd_sweep(arr, extra, seq, cfg) -> None:
    lo, hi = cfg["s6_check"]["qcd_segment"]
    sub, nk = slice_map(arr, extra["nonkf"], lo, hi)
    vo_r = kf_ate(sub.T_cw, sub.kf_frame_idx, seq)
    print(f"[S6 qcd] segment {lo}-{hi}: {len(sub.kf_ids)} keyframes; VO keyframe ATE SE3 {vo_r['se3']['rmse']:.3f} m", flush=True)
    for q in cfg["s6_check"]["qcd_candidates"]:
        c = copy.deepcopy(cfg)
        c["backend"]["stereo_steap"]["qcd"] = [float(q)] * 6
        sp, st = run_steap(sub, nk, seq, c)
        r = kf_ate(sp.poses(), sub.kf_frame_idx, seq)
        print(f"[S6 qcd] qcd={q:g}: keyframe ATE SE3 {r['se3']['rmse']:.3f} m  (obs kf {sp.n_obs_kf} + nonkf "
              f"{sp.n_obs_nonkf}; {st.iterations} it, {st.seconds:.0f} s)", flush=True)


def main() -> None:
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=root / "results" / "s3" / "map_run.npz")
    ap.add_argument("--qcd-sweep", action="store_true")
    ap.add_argument("--qcd", type=float, default=None, help="override backend.stereo_steap.qcd (isotropic)")
    ap.add_argument("--out", type=Path, default=root / "results" / "s6")
    args = ap.parse_args()
    if args.qcd is not None:
        cfg["backend"]["stereo_steap"]["qcd"] = [args.qcd] * 6
    args.out.mkdir(parents=True, exist_ok=True)
    seq = StereoSequence.from_config(cfg)
    arr, extra = load_map(args.map, seq)
    if "nonkf" not in extra:
        raise ValueError(f"{args.map} has no non-keyframe observations; re-run eval/plot_s3_map.py")
    if args.qcd_sweep:
        qcd_sweep(arr, extra, seq, cfg)
        return

    kf_f = arr.kf_frame_idx
    r_vo = kf_ate(arr.T_cw, kf_f, seq)
    # S5 BA on the same map
    ba = build_stereo_ba_problem(arr, seq.rig.K, seq.rig.baseline, cfg["backend"])
    st_ba = solve_lm(ba.problem, cfg["backend"], n_pose=ba.n_pose_dims)
    r_ba = kf_ate(ba.poses(), kf_f, seq)
    print(f"[S6] map {args.map.name}: {len(kf_f)} keyframes. BA: {st_ba.iterations} it {st_ba.seconds:.0f} s", flush=True)
    # S6 STEAP on the same map
    sp, st = run_steap(arr, extra["nonkf"], seq, cfg)
    r_st = kf_ate(sp.poses(), kf_f, seq)
    print(f"[S6] STEAP: landmarks {len(sp.lm_vars)}  obs kf {sp.n_obs_kf} + nonkf {sp.n_obs_nonkf} (stride "
          f"{cfg['backend']['stereo_steap']['nonkeyframe_stride']})  priors {sp.n_prior}  qcd "
          f"{cfg['backend']['stereo_steap']['qcd'][0]:g}  LM {st.iterations} it ({st.termination}) {st.seconds:.0f} s", flush=True)

    # all-frame: VO per-frame poses vs STEAP interpolation, inside the knot span
    f, T_vo, trk = extra["frame_idx"], extra["T_rw"], extra["tracked"]
    t_f = seq.image_ts_ns[f]
    m = trk & seq.gt_valid[f] & (t_f >= arr.kf_t_ns[0]) & (t_f <= arr.kf_t_ns[-1])
    G_all = seq.T_w_rect_gt[f[m], :3, 3]
    T_st_all = sp.interpolate(t_f[m])
    r_vo_all = ate(invert_se3(T_vo[m])[:, :3, 3], G_all, False)
    r_st_all = ate(invert_se3(T_st_all)[:, :3, 3], G_all, False)

    print(f"[S6] keyframe ATE SE3:  VO {r_vo['se3']['rmse']:.3f}  BA {r_ba['se3']['rmse']:.3f}  STEAP {r_st['se3']['rmse']:.3f} m")
    print(f"[S6] keyframe ATE Sim3: VO {r_vo['sim3']['rmse']:.3f}  BA {r_ba['sim3']['rmse']:.3f}  STEAP {r_st['sim3']['rmse']:.3f} m "
          f"(scales {r_vo['sim3']['scale']:.3f} / {r_ba['sim3']['scale']:.3f} / {r_st['sim3']['scale']:.3f})")
    print(f"[S6] all-frame ATE SE3 ({int(m.sum())} frames): VO {r_vo_all['rmse']:.3f}  STEAP (GP-interpolated) {r_st_all['rmse']:.3f} m")

    write_tum(args.out / "steap_all_frames.tum", t_f[m], invert_se3(T_st_all))
    with open(args.out / "summary.json", "w") as fh:
        json.dump({"map": str(args.map), "keyframes": int(len(kf_f)), "qcd": cfg["backend"]["stereo_steap"]["qcd"],
                   "nonkeyframe_stride": cfg["backend"]["stereo_steap"]["nonkeyframe_stride"],
                   "obs_kf": sp.n_obs_kf, "obs_nonkf": sp.n_obs_nonkf, "steap_seconds": st.seconds,
                   "ba_seconds": st_ba.seconds,
                   "kf_ate_se3": {"vo": r_vo["se3"]["rmse"], "ba": r_ba["se3"]["rmse"], "steap": r_st["se3"]["rmse"]},
                   "all_frame_ate_se3": {"vo": r_vo_all["rmse"], "steap": r_st_all["rmse"]}}, fh, indent=2)

    # plot: all-frame trajectories + keyframe BA, error vs time
    gv = seq.gt_valid
    gt = seq.T_w_rect_gt[:, :3, 3]
    t_s = (seq.image_ts_ns - seq.image_ts_ns[0]) * 1e-9
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    for c, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(gt[gv, i], gt[gv, j], c="k", lw=2.2, label="Ground truth")
        ax.plot(r_vo_all["aligned"][:, i], r_vo_all["aligned"][:, j], c="tab:orange", lw=0.9,
                label=f"Stereo VO (all frames), ATE {r_vo_all['rmse']:.3f} m")
        ax.plot(r_ba["se3"]["aligned"][:, i], r_ba["se3"]["aligned"][:, j], c="tab:blue", lw=1.1,
                label=f"Stereo BA (keyframes), ATE {r_ba['se3']['rmse']:.3f} m")
        ax.plot(r_st_all["aligned"][:, i], r_st_all["aligned"][:, j], c="tab:red", lw=1.4,
                label=f"Stereo STEAP (all frames, GP), ATE {r_st_all['rmse']:.3f} m")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if c == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_s[f[m]], r_vo_all["err"], c="tab:orange", lw=0.9, label="VO (all frames)")
    ax.plot(t_s[r_ba["f"]], r_ba["se3"]["err"], c="tab:blue", lw=1.1, label="BA (keyframes)")
    ax.plot(t_s[f[m]], r_st_all["err"], c="tab:red", lw=1.3, label="STEAP (all frames)")
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("position error [m] (each SE3-aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Keyframe ATE SE3: VO {r_vo['se3']['rmse']:.3f}, BA {r_ba['se3']['rmse']:.3f}, STEAP {r_st['se3']['rmse']:.3f} m"
                 f"  |  all-frame: VO {r_vo_all['rmse']:.3f}, STEAP {r_st_all['rmse']:.3f} m")
    fig.suptitle(f"{seq.name}: stereo STEAP (S6) vs stereo BA (S5) vs VO, same VO map")
    fig.tight_layout()
    out = args.out / "s6_steap_vs_ba_vs_vo.png"
    fig.savefig(out, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
