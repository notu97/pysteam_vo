"""S3: stereo VO with a persistent landmark map.

(a) Association check with GT poses (s3_check.gt_segment): landmarks seen in >= min_track_kf
    keyframes are re-triangulated from all their views with GT poses; reprojection in every
    view must be small. Wrong associations show up as large errors.
(b) 5 full-sequence VO runs: coverage, ATE (SE(3), Sim(3)) and RPE over 10 s, compared with the
    S2 runs saved in results/s2 (same metrics), plus an all-runs plot.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_s3_stereo_map [--runs 5] [--skip-gt]
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
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.align import rpe_translation
from pysteam_vo.eval.export_tum import match_to_frames, read_tum, write_tum
from pysteam_vo.eval.run_s2_stereo_vo import ate
from pysteam_vo.frontend.stereo_vo import StereoMapVO
from pysteam_vo.frontend.tracker import triangulate_multiview


def association_check(seq: StereoSequence, cfg: dict) -> dict:
    s3 = cfg["s3_check"]
    a, b = s3["gt_segment"]
    vo = StereoMapVO(seq.rig, cfg)
    T_rw = seq.T_rw_gt
    for i in range(a, b):
        if seq.gt_valid[i]:
            l, r = seq.load_rectified(i)
            vo.process(i, int(seq.image_ts_ns[i]), l, r, T_rw_given=T_rw[i])
    m = vo.map
    K = seq.rig.K
    errs, n_long = [], 0
    for lm in m.landmarks.values():
        if len(lm.obs) < s3["min_track_kf"]:
            continue
        n_long += 1
        Ts, uvs = m.observations(lm.id)
        p = triangulate_multiview(K, Ts, uvs)
        pc = p @ Ts[:, :3, :3].transpose(0, 2, 1) + Ts[:, :3, 3]
        uv = pc[:, :2] / pc[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
        errs.append(np.linalg.norm(uv - uvs, axis=1))
    e = np.concatenate(errs) if errs else np.array([np.inf])
    tl = m.track_lengths()
    return {"segment": [a, b], "keyframes": len(m.keyframes), "landmarks": len(m.landmarks), "long_tracks": n_long,
            "track_len_mean": float(tl.mean()), "track_len_max": int(tl.max()), "views": int(len(e)),
            "reproj_median": float(np.median(e)), "reproj_p90": float(np.percentile(e, 90))}


def score_traj(f: np.ndarray, T_wc: np.ndarray, seq: StereoSequence, rpe_step: int) -> dict:
    ok = seq.gt_valid[f]
    f, T_wc = f[ok], T_wc[ok]
    G = seq.T_w_rect_gt[f]
    r = {"se3": ate(T_wc[:, :3, 3], G[:, :3, 3], False), "sim3": ate(T_wc[:, :3, 3], G[:, :3, 3], True)}
    rpe = rpe_translation(T_wc, G, rpe_step)
    r.update(rpe_median=float(np.median(rpe)), rpe_rmse=float(np.sqrt(np.mean(rpe ** 2))), f=f,
             coverage=float(len(f) / seq.gt_valid.sum()))
    return r


def main() -> None:
    cfg = load_config()
    s3 = cfg["s3_check"]
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=s3["n_runs"])
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--out", type=Path, default=root / "results" / "s3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seq = StereoSequence.from_config(cfg)
    rpe_step = int(round(s3["rpe_window_s"] * 20))                     # EuRoC cam0 is 20 Hz
    summary = {}

    if not args.skip_gt:
        g = association_check(seq, cfg)
        summary["association_gt"] = g
        print(f"[S3a] GT poses, frames {g['segment']}: keyframes {g['keyframes']} landmarks {g['landmarks']} "
              f">= {s3['min_track_kf']} KF: {g['long_tracks']} (track len mean {g['track_len_mean']:.2f}, max "
              f"{g['track_len_max']})  reproj over {g['views']} views: median {g['reproj_median']:.3f} px  "
              f"p90 {g['reproj_p90']:.3f} px", flush=True)

    # ---- S2 baseline from saved runs (same metrics)
    s2 = []
    for p in sorted((root / "results" / "s2").glob("stereo_vo_run*.tum")):
        t, T = read_tum(p)
        f = match_to_frames(t, seq.image_ts_ns, 1_000)
        s2.append(score_traj(f, T, seq, rpe_step))
    if s2:
        print(f"[S2 baseline] {len(s2)} saved runs: ATE SE3 {np.mean([r['se3']['rmse'] for r in s2]):.3f} +- "
              f"{np.std([r['se3']['rmse'] for r in s2]):.3f} m  RPE {s3['rpe_window_s']:.0f}s median "
              f"{np.mean([r['rpe_median'] for r in s2]):.3f} m", flush=True)

    # ---- S3 runs
    runs = []
    for k in range(args.runs):
        vo = StereoMapVO(seq.rig, cfg)
        t0 = time.perf_counter()
        for i in range(len(seq)):
            l, r = seq.load_rectified(i)
            vo.process(i, int(seq.image_ts_ns[i]), l, r)
        secs = time.perf_counter() - t0
        f, T_rw, tracked = vo.trajectory()
        src = np.array([fr.source for fr in vo.frames])
        T_wc = invert_se3(T_rw)
        write_tum(args.out / f"stereo_map_run{k}.tum", seq.image_ts_ns[f[tracked]], T_wc[tracked])
        r = score_traj(f[tracked], T_wc[tracked], seq, rpe_step)
        tl = vo.map.track_lengths()
        r.update(seconds=secs, lost=int((~tracked).sum()), fallback=int((src == "last_keyframe").sum()),
                 keyframes=len(vo.map.keyframes), landmarks=len(vo.map.landmarks), track_mean=float(tl.mean()))
        runs.append(r)
        print(f"[S3b] run {k}: coverage {r['coverage']:.1%}  lost {r['lost']}  fallback-to-last-KF {r['fallback']}  "
              f"KFs {r['keyframes']}  landmarks {r['landmarks']} (track mean {r['track_mean']:.2f})  "
              f"ATE SE3 {r['se3']['rmse']:.3f} m  Sim3 {r['sim3']['rmse']:.3f} (scale {r['sim3']['scale']:.3f})  "
              f"RPE {s3['rpe_window_s']:.0f}s median {r['rpe_median']:.3f} m  {secs:.0f} s", flush=True)

    a3 = np.array([r["se3"]["rmse"] for r in runs])
    p3 = np.array([r["rpe_median"] for r in runs])
    print(f"[S3b] {len(runs)} runs: ATE SE3 {a3.mean():.3f} +- {a3.std():.3f} m  RPE {s3['rpe_window_s']:.0f}s median "
          f"{p3.mean():.3f} +- {p3.std():.3f} m  coverage {np.mean([r['coverage'] for r in runs]):.1%}")
    summary["s3_runs"] = [{k: v for k, v in r.items() if k not in ("se3", "sim3", "f")} |
                          {"ate_se3": r["se3"]["rmse"], "ate_sim3": r["sim3"]["rmse"]} for r in runs]
    if s2:
        summary["s2_runs"] = [{"ate_se3": r["se3"]["rmse"], "rpe_median": r["rpe_median"]} for r in s2]
    with open(args.out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- plot all S3 runs vs GT
    gv = seq.gt_valid
    gt = seq.T_w_rect_gt[:, :3, 3]
    t_s = (seq.image_ts_ns - seq.image_ts_ns[0]) * 1e-9
    colors = plt.cm.tab10(np.arange(len(runs)) % 10)
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    for c, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(gt[gv, i], gt[gv, j], c="k", lw=2.4, label="Ground truth", zorder=10)
        for k, (r, col) in enumerate(zip(runs, colors)):
            ax.plot(r["se3"]["aligned"][:, i], r["se3"]["aligned"][:, j], c=col, lw=1.0, alpha=0.85,
                    label=f"run{k}: ATE {r['se3']['rmse']:.3f} m, coverage {r['coverage']:.1%}")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if c == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for k, (r, col) in enumerate(zip(runs, colors)):
        ax.plot(t_s[r["f"]], r["se3"]["err"], c=col, lw=1.0, label=f"run{k}")
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("position error [m] (SE3-aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=len(runs))
    s2txt = f"; S2: {np.mean([r['se3']['rmse'] for r in s2]):.3f} m" if s2 else ""
    ax.set_title(f"Absolute error vs time: {len(runs)} runs, ATE SE3 {a3.mean():.3f} +- {a3.std():.3f} m{s2txt}")
    fig.suptitle(f"{seq.name}: stereo VO with persistent landmark map (S3), all runs vs ground truth")
    fig.tight_layout()
    fig.savefig(args.out / "s3_all_runs_vs_gt.png", dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    print(f"saved {args.out / 's3_all_runs_vs_gt.png'}")


if __name__ == "__main__":
    main()
