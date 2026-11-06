"""M4 checks for PinholeErrorEvaluator.

1. Analytic Jacobians (pose, landmark) match central finite differences (rtol 1e-5).
2. A single-pose problem with known landmarks recovers the true pose
   (synthetic always; on EuRoC with M2 landmarks triangulated from GT poses).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pylgmath import Transformation, se3op
from pysteam.evaluable import Jacobians
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.stereo import ComposeLandmarkEvaluator, HomoPointStateVar
from pysteam.problem import L2LossFunc, OptimizationProblem
from pysteam.solver import LevMarqGaussNewtonSolver
from scipy.spatial.transform import Rotation

from pysteam_vo.backend.pinhole_error import (
    PinholeErrorEvaluator,
    PinholeIntrinsics,
    pinhole_project,
    reprojection_cost_term,
)
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3

CFG = load_config()
M4 = CFG["m4_check"]
BE = CFG["backend"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
INTS = PinholeIntrinsics.from_K(K)
RNG = np.random.default_rng(4)


def _random_T_cw(rot_scale=0.5, trans_scale=1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(RNG.normal(0, rot_scale, 3)).as_matrix()
    T[:3, 3] = RNG.normal(0, trans_scale, 3)
    return T


def _point_in_front(T_cw: np.ndarray) -> np.ndarray:
    """Random homogeneous world point 2-8 m in front of the camera, inside the image."""
    p_c = np.array([RNG.uniform(-1, 1), RNG.uniform(-0.7, 0.7), 1.0]) * RNG.uniform(2, 8)
    p_w = invert_se3(T_cw) @ np.r_[p_c, 1.0]
    return p_w.reshape(4, 1)


def _error(T_cw: np.ndarray, p_w: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return uv.reshape(2, 1) - pinhole_project(INTS, (T_cw @ p_w).reshape(4, 1))


def _rot_err_deg(R_a, R_b) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_a.T @ R_b).as_rotvec())))


# ---------------------------------------------------------------- Jacobians

@pytest.mark.parametrize("trial", range(M4["n_random_trials"]))
def test_jacobians_match_finite_differences(trial):
    T_cw = _random_T_cw()
    p_w = _point_in_front(T_cw)
    uv = RNG.uniform([0, 0], [752, 480])
    T_var = SE3StateVar(Transformation(T_ba=T_cw.copy()))
    lm_var = HomoPointStateVar(p_w.copy())
    ev = PinholeErrorEvaluator(uv, INTS, ComposeLandmarkEvaluator(T_var, lm_var))

    jacs = Jacobians()
    err = ev.evaluate(np.eye(2), jacs)
    np.testing.assert_allclose(err, _error(T_cw, p_w, uv), rtol=1e-12)
    J_T, J_p = jacs[T_var.key], jacs[lm_var.key]
    assert J_T.shape == (2, 6) and J_p.shape == (2, 3)

    h = M4["fd_step"]
    J_T_fd = np.zeros((2, 6))
    for k in range(6):
        d = np.zeros((6, 1))
        d[k] = h
        e_p = _error(se3op.vec2tran(d) @ T_cw, p_w, uv)        # left perturbation, as SE3StateVar.update
        e_m = _error(se3op.vec2tran(-d) @ T_cw, p_w, uv)
        J_T_fd[:, k] = ((e_p - e_m) / (2 * h)).ravel()
    J_p_fd = np.zeros((2, 3))
    for k in range(3):
        d = np.zeros((4, 1))
        d[k] = h
        J_p_fd[:, k] = ((_error(T_cw, p_w + d, uv) - _error(T_cw, p_w - d, uv)) / (2 * h)).ravel()

    rtol = M4["fd_rtol"]
    np.testing.assert_allclose(J_T, J_T_fd, rtol=rtol, atol=rtol * np.abs(J_T).max())
    np.testing.assert_allclose(J_p, J_p_fd, rtol=rtol, atol=rtol * np.abs(J_p).max())


def test_locked_variables_give_no_jacobians():
    T_cw = _random_T_cw()
    T_var = SE3StateVar(Transformation(T_ba=T_cw), locked=True)
    lm_var = HomoPointStateVar(_point_in_front(T_cw), locked=True)
    ev = PinholeErrorEvaluator(np.zeros(2), INTS, ComposeLandmarkEvaluator(T_var, lm_var))
    assert not ev.active and ev.related_var_keys == set()
    jacs = Jacobians()
    ev.evaluate(np.eye(2), jacs)
    assert len(jacs) == 0


# ---------------------------------------------------------------- single-pose solve

def _solve_pose(T_init: np.ndarray, p_w: np.ndarray, uv: np.ndarray, loss=None) -> np.ndarray:
    """Optimize one camera pose against locked landmarks; return T_cw."""
    T_var = SE3StateVar(Transformation(T_ba=T_init.copy()))
    prob = OptimizationProblem(use_sparse_matrix=True)
    prob.add_state_var(T_var)
    for p, m in zip(p_w, uv):
        lm = HomoPointStateVar(np.r_[p, 1.0].reshape(4, 1), locked=True)
        prob.add_cost_term(reprojection_cost_term(T_var, lm, m, INTS, BE["pixel_sigma_px"], loss=loss,
                                                  cauchy_k=BE["cauchy_k"]))
    LevMarqGaussNewtonSolver(prob, verbose=False, max_iterations=BE["max_iterations"]).optimize()
    return T_var.value.matrix()


def _perturb(T: np.ndarray) -> np.ndarray:
    xi = np.r_[RNG.normal(size=3), RNG.normal(size=3)]
    xi[:3] *= M4["perturb_trans_m"] / np.linalg.norm(xi[:3])
    xi[3:] *= np.radians(M4["perturb_rot_deg"]) / np.linalg.norm(xi[3:])
    return se3op.vec2tran(xi.reshape(6, 1)) @ T


def test_single_pose_synthetic():
    T_true = _random_T_cw(0.3, 0.5)
    p_w = np.stack([_point_in_front(T_true)[:3, 0] for _ in range(150)])
    uv = np.stack([pinhole_project(INTS, (T_true @ np.r_[p, 1.0]).reshape(4, 1)).ravel() for p in p_w])
    uv_noisy = uv + RNG.normal(0, 0.5, uv.shape)
    uv_noisy[:15] += RNG.uniform(30, 80, (15, 2))            # 10% gross outliers, handled by Cauchy
    T_est = _solve_pose(_perturb(T_true), p_w, uv_noisy)
    assert np.linalg.norm(invert_se3(T_est)[:3, 3] - invert_se3(T_true)[:3, 3]) < M4["max_trans_err_m"]
    assert _rot_err_deg(T_est[:3, :3], T_true[:3, :3]) < M4["max_rot_err_deg"]

    T_exact = _solve_pose(_perturb(T_true), p_w, uv, loss=L2LossFunc())   # noise-free: exact
    np.testing.assert_allclose(T_exact, T_true, atol=1e-8)


@needs_data
def test_m4_single_pose_euroc():
    """M4: perturb GT keyframe poses and solve against their M2 (GT-triangulated) landmarks.

    Checks the solver, not GT: independent initializations converge to one pose, that pose
    fits the observations at least as well as GT, and it lies within GT's noise floor.
    """
    from pysteam_vo.eval.plot_m2_map import build_gt_map
    from pysteam_vo.frontend.tracker import project

    seq = EurocSequence.from_config(CFG)
    the_map = build_gt_map(seq, CFG, M4["start_frame"], M4["end_frame"])
    kfs = list(the_map.keyframes.values())
    print()
    n_checked = 0
    for kf in kfs[len(kfs) // 4: 3 * len(kfs) // 4: max(1, len(kfs) // 10)]:
        sel = [(the_map.landmarks[l].p_w, kf.feats.kps[i]) for i, l in enumerate(kf.lm_ids)
               if l >= 0 and len(the_map.landmarks[l].obs) >= M4["min_track_len"]]
        p_w = np.stack([s[0] for s in sel])
        uv = np.stack([s[1] for s in sel])
        T_gt = invert_se3(seq.T_wc_gt[kf.frame_idx])
        ests = [_solve_pose(_perturb(T_gt), p_w, uv) for _ in range(M4["n_inits"])]
        T_est = ests[0]
        spread_m = max(np.linalg.norm(invert_se3(e)[:3, 3] - invert_se3(T_est)[:3, 3]) for e in ests)
        spread_deg = max(_rot_err_deg(e[:3, :3], T_est[:3, :3]) for e in ests)
        dt = float(np.linalg.norm(invert_se3(T_est)[:3, 3] - invert_se3(T_gt)[:3, 3]))
        dr = _rot_err_deg(T_est[:3, :3], T_gt[:3, :3])
        med = lambda T: float(np.median(np.linalg.norm(project(K, T, p_w)[0] - uv, axis=1)))
        r_gt, r_est = med(T_gt), med(T_est)
        print(f"[M4] frame {kf.frame_idx}: {len(p_w)} landmarks  vs GT |dt| {1e3 * dt:.2f} mm "
              f"|dR| {dr:.4f} deg  init spread {1e3 * spread_m:.3f} mm {spread_deg:.1e} deg  "
              f"median reproj GT {r_gt:.3f} px -> solved {r_est:.3f} px")
        assert spread_m < M4["max_init_spread_m"] and spread_deg < M4["max_init_spread_deg"]
        assert r_est <= r_gt
        assert dt < M4["max_trans_err_m"] and dr < M4["max_rot_err_deg"]
        n_checked += 1
    assert n_checked >= 3
