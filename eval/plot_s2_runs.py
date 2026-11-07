"""Overlay every S2 stereo VO run against GT (reads results/s2/stereo_vo_run*.tum).

Each run is SE(3)-aligned to GT on its own tracked frames (stereo has metric scale).

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_s2_runs [--dir pysteam_vo/results/s2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.eval.export_tum import match_to_frames, read_tum
from pysteam_vo.eval.run_s2_stereo_vo import ate


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "s2")
    args = ap.parse_args()
    files = sorted(args.dir.glob("stereo_vo_run*.tum"), key=lambda p: int(p.stem.split("run")[-1]))
    if not files:
        raise FileNotFoundError(f"no stereo_vo_run*.tum in {args.dir}; run eval/run_s2_stereo_vo.py first")
    seq = StereoSequence.from_config(cfg)
    gv = seq.gt_valid
    gt = seq.T_w_rect_gt[:, :3, 3]
    t_s = (seq.image_ts_ns - seq.image_ts_ns[0]) * 1e-9
    path = float(np.sum(np.linalg.norm(np.diff(gt[gv], axis=0), axis=1)))

    res = []
    for p in files:
        t, T_wc = read_tum(p)
        f = match_to_frames(t, seq.image_ts_ns, 1_000)
        ok = (f >= 0) & gv[np.clip(f, 0, None)]
        r = ate(T_wc[ok, :3, 3], gt[f[ok]], with_scale=False)
        r["f"], r["name"], r["cov"] = f[ok], p.stem.replace("stereo_vo_", ""), ok.sum() / gv.sum()
        res.append(r)
    rm = np.array([r["rmse"] for r in res])

    colors = plt.cm.tab10(np.arange(len(res)) % 10)
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    for c, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(gt[gv, i], gt[gv, j], c="k", lw=2.4, label="Ground truth", zorder=10)
        for r, col in zip(res, colors):
            ax.plot(r["aligned"][:, i], r["aligned"][:, j], c=col, lw=1.0, alpha=0.85,
                    label=f"{r['name']}: ATE {r['rmse']:.3f} m, coverage {r['cov']:.1%}")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if c == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for r, col in zip(res, colors):
        ax.plot(t_s[r["f"]], r["err"], c=col, lw=1.0, label=r["name"])
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("position error [m] (SE3-aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=len(res))
    ax.set_title(f"Absolute error vs time: {len(res)} runs, ATE SE3 {rm.mean():.3f} +- {rm.std():.3f} m "
                 f"({100 * rm.mean() / path:.2f}% of {path:.1f} m)")
    fig.suptitle(f"{seq.name}: stereo VO, all {len(res)} runs vs ground truth (each SE(3)-aligned)")
    fig.tight_layout()
    out = args.dir / "s2_all_runs_vs_gt.png"
    fig.savefig(out, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    for r in res:
        print(f"{r['name']}: ATE SE3 {r['rmse']:.3f} m  median {r['median']:.3f}  max {r['max']:.3f}  coverage {r['cov']:.1%}")
    print(f"mean ATE SE3 {rm.mean():.3f} +- {rm.std():.3f} m; saved {out}")


if __name__ == "__main__":
    main()
