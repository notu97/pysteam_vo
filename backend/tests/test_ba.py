"""M5 checks: keyframe bundle adjustment in pysteam (no motion prior).

Synthetic tests always run (seconds). The EuRoC test is marked `slow` (~5 min, pysteam
evaluates every cost term in Python); skip it with `-m "not slow"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.spatial.transform import Rotation

from pysteam_vo.backend.problem import build_ba_problem
from pysteam_vo.backend.solver import SchurLevMarqSolver, solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3
from pysteam_vo.frontend.map import MapArrays
from pysteam_vo.frontend.tracker import project

CFG = load_config()
BE = CFG["backend"]
M5 = CFG["m5_check"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
RNG = np.random.default_rng(5)


def _synthetic_scene(n_kf=8, n_lm=300, noise_px=0.0):
    """Cameras along a gentle arc looking at a box of points; every visible point observed."""
    T_cw = []
    for k in range(n_kf):
        T_wc = np.eye(4)
        T_wc[:3, :3] = Rotation.from_rotvec([0.0, 0.03 * k, 0.0]).as_matrix()
        T_wc[:3, 3] = [0.25 * k, 0.02 * k, 0.0]
        T_cw.append(invert_se3(T_wc))
    T_cw = np.stack(T_cw)
    p_w = RNG.uniform([-2, -1.5, 4], [4, 1.5, 9], size=(n_lm, 3))
    obs_kf, obs_lm, obs_uv = [], [], []
    for k in range(n_kf):
        uv, z = project(K, T_cw[k], p_w)
        vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 752) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        for l in np.flatnonzero(vis):
            obs_kf.append(k)
            obs_lm.append(l)
            obs_uv.append(uv[l] + RNG.normal(0, noise_px, 2) if noise_px else uv[l])
    arr = MapArrays(np.arange(n_kf), np.arange(n_kf), np.arange(n_kf, dtype=np.int64) * 10**8, T_cw,
                    np.arange(n_lm), p_w, np.array(obs_kf), np.array(obs_lm), np.array(obs_uv))
    return arr


def _perturbed(arr: MapArrays, anchor_row: int) -> MapArrays:
    """Perturb every pose but the first and every landmark but the gauge anchor."""
    T = arr.T_cw.copy()
    for k in range(1, len(T)):
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(RNG.normal(0, np.radians(1.0), 3)).as_matrix()
        dT[:3, 3] = RNG.normal(0, 0.03, 3)
        T[k] = dT @ T[k]
    p = arr.p_w + RNG.normal(0, 0.05, arr.p_w.shape)
    p[anchor_row] = arr.p_w[anchor_row]
    return MapArrays(arr.kf_ids, arr.kf_frame_idx, arr.kf_t_ns, T, arr.lm_ids, p,
                     arr.obs_kf, arr.obs_lm, arr.obs_uv)


def test_schur_step_equals_sparse_solve():
    arr = _synthetic_scene(noise_px=0.5)
    ba = build_ba_problem(_perturbed(arr, 0), K, BE)
    solver = SchurLevMarqSolver(ba.problem, ba.n_pose_dims)
    A, b = ba.problem.build_gauss_newton_terms()
    x_schur = solver.solve_gauss_newton(A, b)
    x_ref = spla.spsolve(A.tocsc(), b)[:, None]
    np.testing.assert_allclose(x_schur, x_ref, rtol=1e-6, atol=1e-9 * np.abs(x_ref).max())


def test_gauge_is_fixed():
    arr = _synthetic_scene()
    ba = build_ba_problem(arr, K, BE)
    assert ba.pose_vars[0].locked and not any(v.locked for v in ba.pose_vars[1:])
    assert sum(v.locked for v in ba.lm_vars) == 1
    obs_in_kf0 = set(arr.obs_lm[arr.obs_kf == 0].tolist())
    assert ba.anchor_row in obs_in_kf0


@pytest.mark.parametrize("schur", [True, False])
def test_ba_recovers_noise_free_geometry(schur):
    """Noise-free observations + exact gauge -> BA returns the true poses and points."""
    arr = _synthetic_scene(n_kf=6, n_lm=150)
    ba0 = build_ba_problem(arr, K, BE)                         # just to find the anchor
    ba = build_ba_problem(_perturbed(arr, ba0.anchor_row), K, BE)
    assert ba.anchor_row == ba0.anchor_row
    cfg = dict(BE, relative_cost_change_threshold=1e-12, absolute_cost_change_threshold=1e-14)
    st = solve_lm(ba.problem, cfg, n_pose=ba.n_pose_dims if schur else None)
    assert st.final_cost < 1e-8 * st.initial_cost
    np.testing.assert_allclose(ba.poses(), arr.T_cw, atol=1e-6)
    np.testing.assert_allclose(ba.points(), arr.p_w[ba.lm_rows], atol=1e-5)


def test_ba_reduces_error_with_noise():
    arr = _synthetic_scene(noise_px=0.5)
    ba0 = build_ba_problem(arr, K, BE)
    pert = _perturbed(arr, ba0.anchor_row)
    ba = build_ba_problem(pert, K, BE)
    solve_lm(ba.problem, BE, n_pose=ba.n_pose_dims)
    c = lambda T: invert_se3(T)[:, :3, 3]
    err_before = np.linalg.norm(c(pert.T_cw) - c(arr.T_cw), axis=1).max()
    err_after = np.linalg.norm(c(ba.poses()) - c(arr.T_cw), axis=1).max()
    assert err_after < 0.2 * err_before and err_after < 0.01


# ---------------------------------------------------------------- EuRoC (M5)

@needs_data
@pytest.mark.slow
def test_m5_ba_beats_vo():
    """M5: BA over the VO keyframes lowers the Sim(3) keyframe ATE below VO's (M3)."""
    from pysteam_vo.eval.run_m5_ba import run_m5

    seq = EurocSequence.from_config(CFG)
    r = run_m5(seq, CFG, M5["start_frame"], M5["end_frame"])
    print(f"\n[M5] frames {M5['start_frame']}-{M5['end_frame']}: KFs {r['n_kf']} landmarks {r['n_lm']} "
          f"obs {r['n_obs']}  cost {r['stats'].initial_cost:.0f} -> {r['stats'].final_cost:.0f}  "
          f"{r['stats'].iterations} it ({r['stats'].termination}) {r['stats'].seconds:.0f} s")
    print(f"[M5] keyframe ATE (Sim3): VO {r['ate_vo']['rmse']:.3f} m -> BA {r['ate_ba']['rmse']:.3f} m")
    assert np.isfinite(r["stats"].final_cost) and r["stats"].final_cost < r["stats"].initial_cost
    assert r["ate_ba"]["rmse"] <= M5["max_ate_ratio_vs_vo"] * r["ate_vo"]["rmse"]
