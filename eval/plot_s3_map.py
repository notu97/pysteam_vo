"""One S3 run: stereo VO trajectory + persistent landmark map vs GT (top and side views).

Runs StereoMapVO over the sequence once, saves trajectory + map arrays to results/s3/map_run.npz,
and plots everything in the GT frame using the trajectory's SE(3) alignment (the map gets the
same transform). Long-track landmarks (>= s3_check.min_track_kf keyframes) are highlighted.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_s3_map [--start 0 --end N] [--from-npz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.align import umeyama
from pysteam_vo.frontend.stereo_vo import StereoMapVO


def main() -> None:
    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--from-npz", action="store_true", help="re-plot results/s3/map_run.npz without re-running VO")
    ap.add_argument("--out", type=Path, default=root / "results" / "s3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    npz = args.out / "map_run.npz"
    seq = StereoSequence.from_config(cfg)

    if not args.from_npz:
        end = len(seq) if args.end is None else args.end
        vo = StereoMapVO(seq.rig, cfg)
        for i in range(args.start, end):
            l, r = seq.load_rectified(i)
            vo.process(i, int(seq.image_ts_ns[i]), l, r)
        f, T_rw, tracked = vo.trajectory()
        a = vo.map.to_arrays()
        np.savez_compressed(npz, frame_idx=f, T_rw=T_rw, tracked=tracked, kf_frame_idx=a.kf_frame_idx,
                            kf_T_rw=a.T_cw, lm_p_w=a.p_w, obs_kf=a.obs_kf, obs_lm=a.obs_lm, obs_uv=a.obs_uv,
                            obs_uR=a.obs_uR)
    d = np.load(npz)
    f, T_rw, tracked = d["frame_idx"], d["T_rw"], d["tracked"]
    p_w, kf_T = d["lm_p_w"], d["kf_T_rw"]
    track_len = np.bincount(d["obs_lm"], minlength=len(p_w))

    # SE(3) alignment of the trajectory to GT; same transform for keyframes and landmarks
    use = tracked & seq.gt_valid[f]
    c_est = invert_se3(T_rw[use])[:, :3, 3]
    c_gt = seq.T_w_rect_gt[f[use], :3, 3]
    s, R, t = umeyama(c_est, c_gt, with_scale=False)
    al = lambda P: P @ R.T + t
    traj = al(c_est)
    kf_c = al(invert_se3(kf_T)[:, :3, 3])
    lm = al(p_w)
    err = np.linalg.norm(traj - c_gt, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))

    long_tr = track_len >= cfg["s3_check"]["min_track_kf"]
    pc = cfg["plots"]
    print(f"[S3 map] frames {len(f)} (tracked {tracked.sum()}), keyframes {len(kf_T)}, landmarks {len(p_w)}, "
          f">= {cfg['s3_check']['min_track_kf']} KF: {long_tr.sum()} ({long_tr.mean():.1%}), max track {track_len.max()}, "
          f"stereo obs {np.isfinite(d['obs_uR']).mean():.1%}; ATE SE3 {rmse:.3f} m")

    fig, axs = plt.subplots(1, 2, figsize=(16, 8.5))
    for ax, (i, j, title) in zip(axs, [(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax.scatter(lm[~long_tr, i], lm[~long_tr, j], s=0.6, c="0.72", alpha=0.35, lw=0,
                   label=f"landmarks, 1-2 keyframes ({(~long_tr).sum()})")
        ax.scatter(lm[long_tr, i], lm[long_tr, j], s=1.2, c="tab:blue", alpha=0.6, lw=0,
                   label=f"landmarks, >= {cfg['s3_check']['min_track_kf']} keyframes ({long_tr.sum()})")
        ax.plot(seq.T_w_rect_gt[seq.gt_valid, :3, 3][:, i], seq.T_w_rect_gt[seq.gt_valid, :3, 3][:, j], c="k", lw=2.0,
                label="Ground truth", zorder=4)
        ax.plot(traj[:, i], traj[:, j], c="tab:red", lw=1.3, label=f"Stereo VO (SE3-aligned), ATE {rmse:.3f} m", zorder=5)
        ax.scatter(kf_c[:, i], kf_c[:, j], s=5, c="tab:orange", lw=0, zorder=6, label=f"keyframes ({len(kf_c)})")
        traj_all = np.vstack([traj, c_gt])
        lo, hi = np.percentile(lm[:, [i, j]], [pc["clip_percentile"], 100 - pc["clip_percentile"]], axis=0)
        lo = np.minimum(lo, traj_all[:, [i, j]].min(0)) - pc["pad_m"]
        hi = np.maximum(hi, traj_all[:, [i, j]].max(0)) + pc["pad_m"]
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    leg = axs[0].legend(fontsize=8, loc="best")
    for h in leg.legend_handles[:2]:
        h.set_sizes([20])
    fig.suptitle(f"{seq.name}: stereo VO trajectory + persistent landmark map (S3, one run)")
    fig.tight_layout()
    out = args.out / "s3_vo_and_map.png"
    fig.savefig(out, dpi=pc["dpi"], bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
