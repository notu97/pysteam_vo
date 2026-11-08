"""S5: stereo bundle adjustment on a saved S3 VO map; keyframe ATE vs VO, plot vs GT.

Input: an npz written by eval/plot_s3_map.py (VO trajectory + map of one S3 run). BA starts
from that map's keyframe poses and landmarks (same VO run as S6 will use).

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_s5_stereo_ba [--map pysteam_vo/results/s3/map_run.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.backend.stereo_problem import build_stereo_ba_problem
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.export_tum import write_tum
from pysteam_vo.eval.run_s2_stereo_vo import ate
from pysteam_vo.frontend.map import MapArrays


def load_map(npz: Path, seq: StereoSequence) -> tuple[MapArrays, dict]:
    d = np.load(npz)
    kf_f = d["kf_frame_idx"]
    lm_ids = d["lm_ids"] if "lm_ids" in d else np.arange(len(d["lm_p_w"]))
    arr = MapArrays(np.arange(len(kf_f)), kf_f, seq.image_ts_ns[kf_f], d["kf_T_rw"], lm_ids,
                    d["lm_p_w"], d["obs_kf"], d["obs_lm"], d["obs_uv"], d["obs_uR"])
    extra = {"frame_idx": d["frame_idx"], "T_rw": d["T_rw"], "tracked": d["tracked"]}
    if "nonkf_t_ns" in d:
        extra["nonkf"] = (d["nonkf_t_ns"], d["nonkf_lm_id"], d["nonkf_uv"])
    return arr, extra


def main() -> None:
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=root / "results" / "s3" / "map_run.npz")
    ap.add_argument("--out", type=Path, default=root / "results" / "s5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seq = StereoSequence.from_config(cfg)
    arr, vo = load_map(args.map, seq)

    ba = build_stereo_ba_problem(arr, seq.rig.K, seq.rig.baseline, cfg["backend"])
    assert np.array_equal(ba.poses(), arr.T_cw), "BA must start from the VO keyframe poses"
    print(f"[S5] map {args.map.name}: keyframes {len(arr.kf_ids)}  landmarks used {len(ba.lm_vars)} of {len(arr.lm_ids)} "
          f"(>= {cfg['backend']['stereo_ba']['min_track_kf']} KF)  obs stereo {ba.n_obs_stereo} + mono {ba.n_obs_mono}", flush=True)
    st = solve_lm(ba.problem, cfg["backend"], n_pose=ba.n_pose_dims)
    print(f"[S5] LM: cost {st.initial_cost:.0f} -> {st.final_cost:.0f}  {st.iterations} it ({st.termination})  "
          f"{st.seconds:.0f} s", flush=True)

    # keyframe ATE: VO vs BA on the same keyframes
    kf_f = arr.kf_frame_idx
    ok = seq.gt_valid[kf_f]
    G = seq.T_w_rect_gt[kf_f[ok], :3, 3]
    c_vo = invert_se3(arr.T_cw[ok])[:, :3, 3]
    c_ba = invert_se3(ba.poses()[ok])[:, :3, 3]
    r_vo = {"se3": ate(c_vo, G, False), "sim3": ate(c_vo, G, True)}
    r_ba = {"se3": ate(c_ba, G, False), "sim3": ate(c_ba, G, True)}
    # all-frame VO for reference
    f, T_rw, trk = vo["frame_idx"], vo["T_rw"], vo["tracked"]
    use = trk & seq.gt_valid[f]
    r_vo_all = ate(invert_se3(T_rw[use])[:, :3, 3], seq.T_w_rect_gt[f[use], :3, 3], False)
    path = float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1)))
    print(f"[S5] keyframe ATE SE3:  VO {r_vo['se3']['rmse']:.3f} m  ->  BA {r_ba['se3']['rmse']:.3f} m  "
          f"({100 * (1 - r_ba['se3']['rmse'] / r_vo['se3']['rmse']):+.0f}% reduction; path {path:.1f} m)")
    print(f"[S5] keyframe ATE Sim3: VO {r_vo['sim3']['rmse']:.3f} m (scale {r_vo['sim3']['scale']:.3f})  ->  "
          f"BA {r_ba['sim3']['rmse']:.3f} m (scale {r_ba['sim3']['scale']:.3f})")
    print(f"[S5] VO all-frame ATE SE3 (reference): {r_vo_all['rmse']:.3f} m")

    write_tum(args.out / "ba_keyframes.tum", seq.image_ts_ns[kf_f], invert_se3(ba.poses()))
    with open(args.out / "summary.json", "w") as fh:
        json.dump({"map": str(args.map), "keyframes": int(len(kf_f)), "landmarks_used": len(ba.lm_vars),
                   "obs_stereo": ba.n_obs_stereo, "obs_mono": ba.n_obs_mono, "lm_iterations": st.iterations,
                   "solve_seconds": st.seconds, "ate_kf_se3_vo": r_vo["se3"]["rmse"], "ate_kf_se3_ba": r_ba["se3"]["rmse"],
                   "ate_kf_sim3_vo": r_vo["sim3"]["rmse"], "ate_kf_sim3_ba": r_ba["sim3"]["rmse"],
                   "sim3_scale_ba": r_ba["sim3"]["scale"], "ate_all_se3_vo": r_vo_all["rmse"]}, fh, indent=2)

    # plot
    gv = seq.gt_valid
    gt = seq.T_w_rect_gt[:, :3, 3]
    t_s = (seq.image_ts_ns - seq.image_ts_ns[0]) * 1e-9
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    for c, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(gt[gv, i], gt[gv, j], c="k", lw=2.2, label="Ground truth")
        ax.plot(r_vo["se3"]["aligned"][:, i], r_vo["se3"]["aligned"][:, j], c="tab:orange", lw=1.0,
                label=f"Stereo VO keyframes, ATE {r_vo['se3']['rmse']:.3f} m")
        ax.plot(r_ba["se3"]["aligned"][:, i], r_ba["se3"]["aligned"][:, j], c="tab:red", lw=1.5,
                label=f"Stereo BA keyframes, ATE {r_ba['se3']['rmse']:.3f} m")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if c == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_s[kf_f[ok]], r_vo["se3"]["err"], c="tab:orange", lw=1.0, label="VO")
    ax.plot(t_s[kf_f[ok]], r_ba["se3"]["err"], c="tab:red", lw=1.4, label="BA")
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("keyframe position error [m]\n(each SE3-aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Keyframe ATE (SE3): VO {r_vo['se3']['rmse']:.3f} m -> BA {r_ba['se3']['rmse']:.3f} m "
                 f"({len(kf_f)} keyframes, {len(ba.lm_vars)} landmarks)")
    fig.suptitle(f"{seq.name}: stereo bundle adjustment (S5) vs VO vs ground truth")
    fig.tight_layout()
    out = args.out / "s5_ba_vs_vo_vs_gt.png"
    fig.savefig(out, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
