"""Full-sequence VO vs GT (M9 baseline/comparison).

Runs VO over the whole sequence with automatic restart after tracking loss (as in M8,
including M8's VO overrides), then plots every VO run against GT. Each run has its own
unknown scale, so each run gets its own Sim(3) alignment to GT.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_vo_full [--backend cv2|orbslam3] [--min-run 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.run_m8 import run_vo_segments


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None, help="feature extractor backend (default: config frontend.orb.backend)")
    ap.add_argument("--min-run", type=int, default=50, help="runs with fewer tracked frames are listed but not plotted")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.backend is not None:
        cfg["frontend"]["orb"]["backend"] = args.backend
    backend = cfg["frontend"]["orb"].get("backend", "cv2")
    cfg["frontend"]["vo"].update(cfg["m8"].get("vo_overrides", {}))
    out = args.out or Path(__file__).resolve().parents[1] / "results" / f"vo_full_{backend}.png"

    seq = EurocSequence.from_config(cfg)
    runs, secs = run_vo_segments(seq, cfg, 0, len(seq))
    gt_ok = seq.gt_valid
    t0 = seq.image_ts_ns[0]
    t_s = (seq.image_ts_ns - t0) * 1e-9

    rows, summary = [], {"backend": backend, "vo_seconds": secs, "ms_per_frame": 1e3 * secs / len(seq),
                         "frames_total": int(len(seq)), "frames_with_gt": int(gt_ok.sum()), "runs": []}
    for r in runs:
        f, T_cw = r.tracked_poses()
        if len(f) == 0:
            continue
        ok = gt_ok[f]
        info = {"first": int(f[0]), "last": int(f[-1]), "tracked": int(len(f)), "keyframes": len(r.map.keyframes),
                "end_state": r.state}
        if ok.sum() >= 3:
            a = ate_sim3(invert_se3(T_cw[ok])[:, :3, 3], seq.T_wc_gt[f[ok], :3, 3])
            info.update(ate_rmse=a["rmse"], path_m=float(np.sum(np.linalg.norm(np.diff(seq.T_wc_gt[f[ok], :3, 3], axis=0), axis=1))))
            rows.append((f[ok], a))
        summary["runs"].append(info)
    tracked = np.zeros(len(seq), bool)
    for r in runs:
        f, _ = r.tracked_poses()
        tracked[f] = True
    longest = max(summary["runs"], key=lambda x: x["tracked"])
    summary.update(coverage_any_run=float(tracked[gt_ok].mean()),
                   coverage_longest_run=float(longest["tracked"] / gt_ok.sum()),
                   n_runs=len(summary["runs"]), n_losses=sum(r["end_state"] == "lost" for r in summary["runs"]))

    print(f"[VO full] backend={backend}  {secs:.0f} s ({summary['ms_per_frame']:.1f} ms/frame)  runs {summary['n_runs']}  "
          f"tracking losses {summary['n_losses']}")
    for x in summary["runs"]:
        ate = f"ATE {x['ate_rmse']:.3f} m over {x['path_m']:.1f} m" if "ate_rmse" in x else "no GT"
        print(f"  run frames {x['first']:4d}-{x['last']:4d}: tracked {x['tracked']:4d}, keyframes {x['keyframes']:3d}, "
              f"end {x['end_state']:8s} {ate}")
    print(f"  coverage of GT frames: any run {summary['coverage_any_run']:.1%}, longest run {summary['coverage_longest_run']:.1%}")

    # ---- plot
    gt = seq.T_wc_gt[:, :3, 3]
    big = [(f, a) for f, a in rows if len(f) >= args.min_run]
    colors = plt.cm.tab10(np.arange(max(1, len(big))) % 10)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 0.8, 0.35])
    for k, (i, j, title) in enumerate([(0, 1, "Top view (x-y)"), (0, 2, "Side view (x-z)")]):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(gt[gt_ok, i], gt[gt_ok, j], c="k", lw=2.0, label="Ground truth (all frames)")
        for (f, a), c in zip(big, colors):
            ax.plot(a["aligned"][:, i], a["aligned"][:, j], c=c, lw=1.2,
                    label=f"VO run {f[0]}-{f[-1]} (ATE {a['rmse']:.3f} m)")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    ax = fig.add_subplot(gs[1, :])
    for (f, a), c in zip(big, colors):
        ax.plot(t_s[f], a["err"], c=c, lw=1.0)
    ax.set_ylabel("position error [m]\n(per-run Sim3)")
    ax.grid(alpha=0.3)
    ax.set_title("Absolute error vs time (each run aligned separately)")
    axt = fig.add_subplot(gs[2, :], sharex=ax)
    axt.fill_between(t_s, 0, gt_ok.astype(float), step="mid", color="0.85", label="GT available")
    axt.fill_between(t_s, 0, tracked.astype(float) * 0.6, step="mid", color="tab:green", label="VO tracked")
    for x in summary["runs"]:
        if x["end_state"] == "lost":
            axt.axvline(t_s[x["last"]], c="tab:red", lw=1)
    axt.set_yticks([])
    axt.set_xlabel("time since first frame [s]")
    axt.legend(fontsize=8, loc="upper right", ncol=2)
    axt.set_title(f"Coverage (red lines = tracking lost): any run {summary['coverage_any_run']:.0%}, "
                  f"longest run {summary['coverage_longest_run']:.0%}", fontsize=10)
    fig.suptitle(f"{seq.name}: full-sequence VO ({backend} extractor) vs ground truth")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
