"""S2: full-sequence stereo VO vs GT, repeated runs, plus the reference script's trajectory.

Coverage = frames successfully tracked (PnP succeeded) / frames with GT.
ATE (SE(3), primary) and ATE (Sim(3)) on tracked frames, rectified-left camera positions vs GT.
The reference (Orb-feat-python/vo_euroc.py output, body frame) is scored on body positions.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_s2_stereo_vo [--runs 5] [--start 0 --end N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import interpolate_poses, invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.align import umeyama
from pysteam_vo.eval.export_tum import match_to_frames, read_tum, write_tum
from pysteam_vo.frontend.stereo_vo import StereoVO


def ate(p_est: np.ndarray, p_gt: np.ndarray, with_scale: bool) -> dict:
    s, R, t = umeyama(p_est, p_gt, with_scale=with_scale)
    al = s * p_est @ R.T + t
    e = np.linalg.norm(al - p_gt, axis=1)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "median": float(np.median(e)), "max": float(e.max()),
            "scale": float(s), "err": e, "aligned": al}


def run_once(seq: StereoSequence, cfg: dict, start: int, end: int) -> tuple[StereoVO, float]:
    vo = StereoVO(seq.rig, cfg)
    t0 = time.perf_counter()
    for i in range(start, end):
        l, r = seq.load_rectified(i)
        vo.process(i, int(seq.image_ts_ns[i]), l, r)
    return vo, time.perf_counter() - t0


def score(vo: StereoVO, seq: StereoSequence) -> dict:
    f, T_rw, tracked = vo.trajectory()
    gt_f = seq.gt_valid[f]
    use = tracked & gt_f
    p_est = invert_se3(T_rw[use])[:, :3, 3]
    p_gt = seq.T_w_rect_gt[f[use], :3, 3]
    path = float(np.sum(np.linalg.norm(np.diff(seq.T_w_rect_gt[f[gt_f], :3, 3], axis=0), axis=1)))
    lost = np.flatnonzero(~tracked)
    return {"frames": int(len(f)), "tracked": int(tracked.sum()), "lost": int(len(lost)),
            "coverage": float(use.sum() / seq.gt_valid[f[0]:f[-1] + 1].sum()),
            "coverage_of_all_gt": float(use.sum() / seq.gt_valid.sum()),
            "keyframes": len(vo.keyframes), "path_m": path, "lost_frames": f[lost].tolist(),
            "se3": ate(p_est, p_gt, False), "sim3": ate(p_est, p_gt, True), "frame_idx": f[use]}


def score_reference(path: Path, seq: StereoSequence) -> dict | None:
    if not path.is_file():
        return None
    t, T_wb = read_tum(path)
    f = match_to_frames(t, seq.image_ts_ns, 1_000_000)
    ok = f >= 0
    T_wb_gt, gv = interpolate_poses(seq.gt_ts_ns, seq.T_wb_gt, seq.image_ts_ns[f[ok]], seq.gt_max_gap_ns)
    p, g = T_wb[ok][gv, :3, 3], T_wb_gt[gv, :3, 3]
    return {"frames": int(ok.sum()), "se3": ate(p, g, False), "sim3": ate(p, g, True)}


def main() -> None:
    cfg = load_config()
    s2 = cfg["s2_check"]
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=s2["n_runs"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", type=Path, default=root / "results" / "s2")
    args = ap.parse_args()
    seq = StereoSequence.from_config(cfg)
    end = len(seq) if args.end is None else args.end
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for k in range(args.runs):
        vo, secs = run_once(seq, cfg, args.start, end)
        r = score(vo, seq)
        r["seconds"] = secs
        runs.append((vo, r))
        print(f"[S2] run {k}: frames {r['frames']}  tracked {r['tracked']}  lost {r['lost']}  keyframes {r['keyframes']}  "
              f"coverage {r['coverage']:.1%}  ATE SE3 {r['se3']['rmse']:.3f} m ({100*r['se3']['rmse']/r['path_m']:.2f}% of "
              f"{r['path_m']:.1f} m)  Sim3 {r['sim3']['rmse']:.3f} m (scale {r['sim3']['scale']:.3f})  "
              f"{secs:.0f} s ({1e3*secs/r['frames']:.1f} ms/frame)", flush=True)
    se3 = np.array([r["se3"]["rmse"] for _, r in runs])
    sim3 = np.array([r["sim3"]["rmse"] for _, r in runs])
    cov = np.array([r["coverage"] for _, r in runs])
    print(f"[S2] {len(runs)} runs: coverage {cov.mean():.1%} +- {cov.std():.1%} | ATE SE3 {se3.mean():.3f} +- {se3.std():.3f} m "
          f"(min {se3.min():.3f}, max {se3.max():.3f}) | ATE Sim3 {sim3.mean():.3f} +- {sim3.std():.3f} m")
    ref = score_reference((root / s2["reference_tum"]).resolve(), seq)
    if ref:
        print(f"[S2] reference vo_euroc.py (body frame, single run): ATE SE3 {ref['se3']['rmse']:.3f} m  "
              f"Sim3 {ref['sim3']['rmse']:.3f} m (scale {ref['sim3']['scale']:.3f})")

    # ---- export every run (for eval/plot_s2_runs.py), then plot the median-ATE run
    for k, (vo_k, _) in enumerate(runs):
        fk, Tk, trk = vo_k.trajectory()
        write_tum(args.out / f"stereo_vo_run{k}.tum", seq.image_ts_ns[fk[trk]], invert_se3(Tk[trk]))
    k_med = int(np.argsort(se3)[len(se3) // 2])
    vo, r = runs[k_med]
    f, T_rw, tracked = vo.trajectory()
    write_tum(args.out / "stereo_vo.tum", seq.image_ts_ns[f], invert_se3(T_rw))
    gv = seq.gt_valid
    write_tum(args.out / "gt_rect.tum", seq.image_ts_ns[gv], seq.T_w_rect_gt[gv])
    summary = {"runs": [{k: v for k, v in rr.items() if k not in ("se3", "sim3", "frame_idx")} |
                        {"ate_se3": rr["se3"]["rmse"], "ate_sim3": rr["sim3"]["rmse"], "sim3_scale": rr["sim3"]["scale"]}
                        for _, rr in runs],
               "mean": {"coverage": float(cov.mean()), "ate_se3": float(se3.mean()), "ate_se3_std": float(se3.std()),
                        "ate_sim3": float(sim3.mean())},
               "reference": None if not ref else {"ate_se3": ref["se3"]["rmse"], "ate_sim3": ref["sim3"]["rmse"]},
               "plotted_run": k_med}
    with open(args.out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    t_s = (seq.image_ts_ns - seq.image_ts_ns[0]) * 1e-9
    gt = seq.T_w_rect_gt[:, :3, 3]
    al = r["se3"]["aligned"]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 0.8, 0.35])
    for c, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(gt[gv, i], gt[gv, j], c="k", lw=2.0, label="Ground truth")
        ax.plot(al[:, i], al[:, j], c="tab:red", lw=1.2, label=f"Stereo VO (SE3-aligned), ATE {r['se3']['rmse']:.3f} m")
        ax.scatter(gt[gv][0, i], gt[gv][0, j], c="tab:green", s=50, zorder=5, label="start")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if c == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_s[r["frame_idx"]], r["se3"]["err"], c="tab:red", lw=1.0)
    ax.set_ylabel("position error [m]\n(SE3-aligned)")
    ax.grid(alpha=0.3)
    ax.set_title(f"Absolute error vs time (run {k_med}; {len(runs)}-run ATE SE3 {se3.mean():.3f} +- {se3.std():.3f} m)")
    axt = fig.add_subplot(gs[2, :], sharex=ax)
    trk = np.zeros(len(seq), bool)
    trk[f[tracked]] = True
    axt.fill_between(t_s, 0, gv.astype(float), step="mid", color="0.85", label="GT available")
    axt.fill_between(t_s, 0, trk.astype(float) * 0.6, step="mid", color="tab:green", label="tracked")
    for i in f[~tracked]:
        axt.axvline(t_s[i], c="tab:red", lw=0.8)
    axt.set_yticks([])
    axt.set_xlabel("time since first frame [s]")
    axt.legend(fontsize=8, loc="upper right", ncol=2)
    axt.set_title(f"Coverage {r['coverage']:.1%} (red = PnP failed, pose from motion model; {r['lost']} frames)", fontsize=10)
    fig.suptitle(f"{seq.name}: stereo VO (rectified cam0/cam1, ORB-SLAM3 extractor) vs ground truth")
    fig.tight_layout()
    fig.savefig(args.out / "s2_stereo_vo_vs_gt.png", dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    print(f"saved {args.out / 's2_stereo_vo_vs_gt.png'}")


if __name__ == "__main__":
    main()
