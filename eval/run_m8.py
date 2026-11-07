"""M8: full-sequence STEAP export + comparison with VGGT-Omega on identical timestamps.

1. VO (M3) over the whole sequence with automatic restart after tracking loss; the longest
   continuously tracked run is used. STEAP (M6) is initialized from that run's VO map.
2. Export a GP-interpolated pose for every cam0 frame inside the knot span (steap.tum),
   plus VO per-frame poses (vo.tum) and GT camera poses (gt_cam.tum). World-from-camera.
3. Common evaluation set: VGGT-Omega's frames (every 5th cam0 frame) that STEAP also
   covers and that have GT. Written to common/{gt,steap,vggt,vo}.tum with identical,
   exact timestamps; scored with eval/run_evo.sh.

Run from ORB_based_pysteam/:
    python -m pysteam_vo.eval.run_m8 [--vggt-dir ../out_mh01_s5_c60_o20_r512_undist] [--out pysteam_vo/results/m8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pysteam_vo.backend.problem import build_steap_problem
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.eval.export_tum import match_to_frames, read_tum, write_tum
from pysteam_vo.frontend.tracker import MonoVO


def run_vo_segments(seq: EurocSequence, cfg: dict, start: int, end: int) -> tuple[list[MonoVO], float]:
    """VO with automatic restart: when tracking is lost, a fresh MonoVO (new map, new scale)
    starts on the next frame. Returns every run and the total VO time. No sequence-specific
    start frame is needed; M8 then uses the longest continuously tracked run.
    """
    runs = [MonoVO(seq.calib, cfg["frontend"])]
    t0 = time.perf_counter()
    for i in range(start, end):
        if runs[-1].state == "lost":
            runs.append(MonoVO(seq.calib, cfg["frontend"]))
        runs[-1].process(i, int(seq.image_ts_ns[i]), seq.load_image(i))
    return runs, time.perf_counter() - t0


def main() -> None:
    cfg = load_config()
    m8 = cfg["m8"]
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--vggt-dir", type=Path, default=(root / m8["vggt_dir"]).resolve())
    ap.add_argument("--out", type=Path, default=root / "results" / "m8")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg["frontend"]["vo"].update(m8.get("vo_overrides", {}))
    print(f"[M8] VO overrides: {m8.get('vo_overrides', {})}", flush=True)
    seq = EurocSequence.from_config(cfg)
    end = len(seq) if args.end is None else args.end
    ts = seq.image_ts_ns

    # ---- 1. VO + STEAP (initialized from VO)
    runs, vo_secs = run_vo_segments(seq, cfg, args.start, end)
    runs_info = []
    for r in runs:
        f, _ = r.tracked_poses()
        runs_info.append({"first": int(f[0]) if len(f) else None, "last": int(f[-1]) if len(f) else None,
                          "tracked": int(len(f)), "keyframes": len(r.map.keyframes), "state": r.state})
        print(f"[M8] VO run: frames {runs_info[-1]['first']}-{runs_info[-1]['last']}  tracked {len(f)}  "
              f"keyframes {len(r.map.keyframes)}  end state {r.state}", flush=True)
    vo = max(runs, key=lambda r: len(r.tracked_poses()[0]))
    arr = vo.map.to_arrays()
    nonkf = vo.nonkeyframe_observations()
    print(f"[M8] using longest VO run: keyframes {len(arr.kf_ids)} landmarks {len(arr.lm_ids)} "
          f"(all VO {vo_secs:.0f} s)", flush=True)
    t0 = time.perf_counter()
    sp = build_steap_problem(arr, nonkf, seq.calib.K, cfg["backend"])
    assert np.array_equal(sp.poses(), arr.T_cw), "STEAP must start from the VO keyframe poses"
    build_secs = time.perf_counter() - t0
    st = solve_lm(sp.problem, cfg["backend"], n_pose=sp.n_pose_dims)
    print(f"[M8] STEAP: knots {len(sp.pose_vars)} obs kf {sp.n_obs_kf} + non-kf {sp.n_obs_nonkf}; "
          f"LM {st.iterations} it ({st.termination}) {st.seconds:.0f} s", flush=True)

    # ---- 2. per-frame export
    span = (ts >= arr.kf_t_ns[0]) & (ts <= arr.kf_t_ns[-1])
    f_steap = np.flatnonzero(span)
    T_wc_steap = invert_se3(sp.interpolate(ts[f_steap]))
    f_vo, T_cw_vo = vo.tracked_poses()
    f_gt = np.flatnonzero(seq.gt_valid)
    write_tum(out / "steap.tum", ts[f_steap], T_wc_steap)
    write_tum(out / "vo.tum", ts[f_vo], invert_se3(T_cw_vo))
    write_tum(out / "gt_cam.tum", ts[f_gt], seq.T_wc_gt[f_gt])
    np.savez_compressed(out / "steap_solution.npz", kf_t_ns=arr.kf_t_ns, kf_T_cw=sp.poses(),
                        kf_w=sp.velocities(), lm_p_w=sp.points(), frame_idx=f_steap, T_wc=T_wc_steap)

    # ---- 3. common set with VGGT-Omega
    t_vg, T_vg = read_tum(args.vggt_dir / "traj_est_tum.txt")
    f_vg = match_to_frames(t_vg, ts, int(m8["match_tol_ns"]))
    assert (f_vg >= 0).all(), "some VGGT-Omega stamps do not match a cam0 frame"
    steap_row = -np.ones(len(ts), dtype=np.int64)
    steap_row[f_steap] = np.arange(len(f_steap))
    vo_row = -np.ones(len(ts), dtype=np.int64)
    vo_row[f_vo] = np.arange(len(f_vo))
    common = (steap_row[f_vg] >= 0) & (vo_row[f_vg] >= 0) & seq.gt_valid[f_vg]
    fc = f_vg[common]
    cdir = out / "common"
    write_tum(cdir / "gt.tum", ts[fc], seq.T_wc_gt[fc])
    write_tum(cdir / "steap.tum", ts[fc], T_wc_steap[steap_row[fc]])
    write_tum(cdir / "vggt.tum", ts[fc], T_vg[common])
    write_tum(cdir / "vo.tum", ts[fc], invert_se3(T_cw_vo[vo_row[fc]]))

    # ---- python-side sanity numbers (evo is the reported metric)
    gt_all_vg = seq.gt_valid[f_vg]
    c = lambda T: T[:, :3, 3]
    res = {
        "sequence": seq.name, "frames_total": int(len(ts)), "frames_with_gt": int(seq.gt_valid.sum()),
        "steap_frames": int(len(f_steap)), "vo_tracked_frames": int(len(f_vo)), "vo_runs": runs_info,
        "steap_coverage_of_gt_frames": float(np.mean(span[seq.gt_valid])),
        "vggt_frames": int(len(f_vg)), "common_frames": int(len(fc)),
        "vggt_frames_with_gt": int(gt_all_vg.sum()),
        "knots": len(sp.pose_vars), "landmarks": len(sp.lm_vars),
        "obs_kf": sp.n_obs_kf, "obs_nonkf": sp.n_obs_nonkf, "lm_iterations": st.iterations,
        "runtime_s": {"vo": vo_secs, "steap_build": build_secs, "steap_solve": st.seconds},
        "ate_common_python": {
            k: {"rmse": ate_sim3(c(T), c(seq.T_wc_gt[fc]))["rmse"], "scale": ate_sim3(c(T), c(seq.T_wc_gt[fc]))["scale"]}
            for k, T in [("steap", T_wc_steap[steap_row[fc]]), ("vggt", T_vg[common]),
                         ("vo", invert_se3(T_cw_vo[vo_row[fc]]))]
        },
        "ate_vggt_all_its_frames_python": ate_sim3(c(T_vg[gt_all_vg]), c(seq.T_wc_gt[f_vg[gt_all_vg]]))["rmse"],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
