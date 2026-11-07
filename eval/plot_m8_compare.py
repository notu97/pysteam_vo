"""M8 plot: STEAP vs VGGT-Omega vs GT on identical timestamps (results/m8/common).

Each estimate gets its own Sim(3) alignment to GT (same as `evo_ape -as`).

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.plot_m8_compare [--dir pysteam_vo/results/m8] [--no-vo]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysteam_vo.config import load_config
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.export_tum import read_tum


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=root / "results" / "m8")
    ap.add_argument("--no-vo", action="store_true")
    args = ap.parse_args()
    cdir = args.dir / "common"
    dpi = load_config()["plots"]["dpi"]

    t, T_gt = read_tum(cdir / "gt.tum")
    p_gt = T_gt[:, :3, 3]
    methods = [("steap", "STEAP (ours)", "tab:red", 1.6), ("vggt", "VGGT-Omega", "tab:blue", 1.3)]
    if not args.no_vo:
        methods.append(("vo", "VO (front-end only)", "tab:orange", 0.9))
    res = {}
    for key, *_ in methods:
        t_m, T_m = read_tum(cdir / f"{key}.tum")
        assert np.allclose(t_m, t), f"{key}.tum timestamps differ from gt.tum"
        res[key] = ate_sim3(T_m[:, :3, 3], p_gt)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    views = [(fig.add_subplot(gs[0, 0]), 0, 1, "Top view (x-y)"), (fig.add_subplot(gs[0, 1]), 0, 2, "Side view (x-z)")]
    for ax, i, j, title in views:
        ax.plot(p_gt[:, i], p_gt[:, j], c="k", lw=2.2, label="Ground truth")
        for key, lab, col, lw in methods:
            a = res[key]["aligned"]
            ax.plot(a[:, i], a[:, j], c=col, lw=lw, label=f"{lab}  ATE {res[key]['rmse']:.3f} m")
        ax.scatter(p_gt[0, i], p_gt[0, j], s=50, c="tab:green", zorder=5, label="start")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    views[0][0].legend(fontsize=8, loc="best")

    ax = fig.add_subplot(gs[1, :])
    ts = t - t[0]
    for key, lab, col, lw in methods:
        ax.plot(ts, res[key]["err"], c=col, lw=1.0, label=f"{lab}  (RMSE {res[key]['rmse']:.3f} m, "
                                                             f"median {res[key]['median']:.3f} m)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error after Sim(3) [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Absolute trajectory error vs time")

    fig.suptitle(f"EuRoC MH_01_easy cam0: STEAP vs VGGT-Omega on {len(t)} identical timestamps "
                 f"(every 5th frame, {ts[-1]:.0f} s covered by STEAP)")
    fig.tight_layout()
    out = args.dir / "m8_steap_vs_vggt.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    for key, lab, *_ in methods:
        print(f"{lab:22s} ATE rmse {res[key]['rmse']:.4f} m  median {res[key]['median']:.4f} m  "
              f"max {res[key]['max']:.4f} m  scale {res[key]['scale']:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
