"""S4 checks for RectStereoErrorEvaluator (mirrors M4).

1. The model equals the S1 rig model (StereoRig.project).
2. Analytic Jacobians (pose, landmark) match central finite differences (rtol 1e-5).
3. Single-pose solves recover the pose (synthetic; EuRoC with GT-posed stereo landmarks).
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

from pysteam_vo.backend.stereo_error import RectStereoErrorEvaluator, stereo_cost_term, stereo_intrinsics
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3, load_camera_calib
from pysteam_vo.data.stereo import StereoRig, StereoSequence

CFG = load_config()
S4 = CFG["s4_check"]
BE = CFG["backend"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
RNG = np.random.default_rng(44)
# Rectified EuRoC-like intrinsics for the synthetic tests (real ones come from the rig).
K = np.array([[435.2, 0.0, 367.5], [0.0, 435.2, 252.2], [0.0, 0.0, 1.0]])
B = 0.110
INTS = stereo_intrinsics(K, B)


def _random_T(rot_scale=0.5, trans_scale=1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(RNG.normal(0, rot_scale, 3)).as_matrix()
    T[:3, 3] = RNG.normal(0, trans_scale, 3)
    return T


def _point_in_front(T_rw: np.ndarray) -> np.ndarray:
    p_c = np.array([RNG.uniform(-1, 1), RNG.uniform(-0.7, 0.7), 1.0]) * RNG.uniform(1.5, 10)
    return (invert_se3(T_rw) @ np.r_[p_c, 1.0]).reshape(4, 1)


def _obs(T_rw, p_w):
    p_c = T_rw @ p_w.reshape(4)
    x, y, z = p_c[:3]
    return np.array([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2], K[0, 0] * (x - B) / z + K[0, 2]])


def _err(T_rw, p_w, meas):
    return meas.reshape(3, 1) - _obs(T_rw, p_w).reshape(3, 1)


def _rot_err_deg(Ra, Rb) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


# ---------------------------------------------------------------- model + Jacobians

@needs_data
def test_model_equals_rig_projection():
    m = SEQ_DIR / "mav0"
    rig = StereoRig.from_calibs(load_camera_calib(m / "cam0" / "sensor.yaml"), load_camera_calib(m / "cam1" / "sensor.yaml"))
    ints = stereo_intrinsics(rig.K, rig.baseline)
    T = _random_T()
    for _ in range(20):
        p_w = _point_in_front(T)
        ev = RectStereoErrorEvaluator(np.zeros(3), ints, ComposeLandmarkEvaluator(
            SE3StateVar(Transformation(T_ba=T), locked=True), HomoPointStateVar(p_w.copy(), locked=True)))
        obs_rig, _ = rig.project((T @ p_w.reshape(4))[None, :3])
        np.testing.assert_allclose(-ev.evaluate().ravel(), obs_rig[0], rtol=1e-12)


@pytest.mark.parametrize("trial", range(S4["n_random_trials"]))
def test_jacobians_match_finite_differences(trial):
    T = _random_T()
    p_w = _point_in_front(T)
    meas = _obs(T, p_w) + RNG.normal(0, 2.0, 3)
    T_var = SE3StateVar(Transformation(T_ba=T.copy()))
    lm_var = HomoPointStateVar(p_w.copy())
    ev = RectStereoErrorEvaluator(meas, INTS, ComposeLandmarkEvaluator(T_var, lm_var))
    jacs = Jacobians()
    err = ev.evaluate(np.eye(3), jacs)
    np.testing.assert_allclose(err, _err(T, p_w, meas), rtol=1e-12, atol=1e-12)
    J_T, J_p = jacs[T_var.key], jacs[lm_var.key]
    assert J_T.shape == (3, 6) and J_p.shape == (3, 3)
    h = S4["fd_step"]
    J_T_fd = np.zeros((3, 6))
    for k in range(6):
        d = np.zeros((6, 1))
        d[k] = h
        J_T_fd[:, k] = ((_err(se3op.vec2tran(d) @ T, p_w, meas) - _err(se3op.vec2tran(-d) @ T, p_w, meas)) / (2 * h)).ravel()
    J_p_fd = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros((4, 1))
        d[k] = h
        J_p_fd[:, k] = ((_err(T, p_w + d, meas) - _err(T, p_w - d, meas)) / (2 * h)).ravel()
    rtol = S4["fd_rtol"]
    np.testing.assert_allclose(J_T, J_T_fd, rtol=rtol, atol=rtol * np.abs(J_T).max())
    np.testing.assert_allclose(J_p, J_p_fd, rtol=rtol, atol=rtol * np.abs(J_p).max())


def test_locked_variables_give_no_jacobians():
    T = _random_T()
    ev = RectStereoErrorEvaluator(np.zeros(3), INTS, ComposeLandmarkEvaluator(
        SE3StateVar(Transformation(T_ba=T), locked=True), HomoPointStateVar(_point_in_front(T), locked=True)))
    assert not ev.active and ev.related_var_keys == set()
    jacs = Jacobians()
    ev.evaluate(np.eye(3), jacs)
    assert len(jacs) == 0


# ---------------------------------------------------------------- single-pose solves

def _solve_pose(T_init, p_w, obs, ints, loss=None):
    T_var = SE3StateVar(Transformation(T_ba=T_init.copy()))
    prob = OptimizationProblem(use_sparse_matrix=True)
    prob.add_state_var(T_var)
    for p, o in zip(p_w, obs):
        lm = HomoPointStateVar(np.r_[p, 1.0].reshape(4, 1), locked=True)
        prob.add_cost_term(stereo_cost_term(T_var, lm, o, ints, BE["pixel_sigma_px"], loss=loss, cauchy_k=BE["cauchy_k"]))
    LevMarqGaussNewtonSolver(prob, verbose=False, max_iterations=BE["max_iterations"]).optimize()
    return T_var.value.matrix()


def _perturb(T):
    xi = np.r_[RNG.normal(size=3), RNG.normal(size=3)]
    xi[:3] *= S4["perturb_trans_m"] / np.linalg.norm(xi[:3])
    xi[3:] *= np.radians(S4["perturb_rot_deg"]) / np.linalg.norm(xi[3:])
    return se3op.vec2tran(xi.reshape(6, 1)) @ T


def test_single_pose_synthetic():
    T_true = _random_T(0.3, 0.5)
    p_w = np.stack([_point_in_front(T_true)[:3, 0] for _ in range(150)])
    obs = np.stack([_obs(T_true, np.r_[p, 1.0]) for p in p_w])
    noisy = obs + RNG.normal(0, 0.5, obs.shape)
    noisy[:15] += RNG.uniform(30, 80, (15, 3))                       # 10% gross outliers
    T_est = _solve_pose(_perturb(T_true), p_w, noisy, INTS)
    assert np.linalg.norm(invert_se3(T_est)[:3, 3] - invert_se3(T_true)[:3, 3]) < 0.005
    assert _rot_err_deg(T_est[:3, :3], T_true[:3, :3]) < 0.05
    np.testing.assert_allclose(_solve_pose(_perturb(T_true), p_w, obs, INTS, loss=L2LossFunc()), T_true, atol=1e-8)


@needs_data
def test_s4_single_pose_euroc():
    """Perturb GT keyframe poses and solve against GT-posed stereo landmarks (stereo observations)."""
    from pysteam_vo.frontend.stereo_vo import StereoMapVO
    from pysteam_vo.frontend.tracker import triangulate_multiview

    seq = StereoSequence.from_config(CFG)
    rig = seq.rig
    ints = stereo_intrinsics(rig.K, rig.baseline)
    a, b = S4["gt_segment"]
    vo = StereoMapVO(rig, CFG)
    for i in range(a, b):
        l, r = seq.load_rectified(i)
        vo.process(i, int(seq.image_ts_ns[i]), l, r, T_rw_given=seq.T_rw_gt[i])
    m = vo.map
    # landmark positions re-triangulated from all views with GT poses ("GT landmarks")
    p_ref = {}
    for lm in m.landmarks.values():
        if len(lm.obs) >= S4["min_track_kf"]:
            Ts, uvs = m.observations(lm.id)
            p_ref[lm.id] = triangulate_multiview(rig.K, Ts, uvs)
    kfs = list(m.keyframes.values())
    k_res, sig = BE["cauchy_k"], BE["pixel_sigma_px"]
    print()
    n_checked = 0
    for kf in kfs[len(kfs) // 4: 3 * len(kfs) // 4: max(1, len(kfs) // 10)]:
        sel = [(p_ref[l], np.r_[kf.feats.kps[i], kf.uR[i]]) for i, l in enumerate(kf.lm_ids)
               if l >= 0 and l in p_ref and np.isfinite(kf.uR[i])]
        if len(sel) < S4["min_landmarks"]:
            continue
        p_w = np.stack([s[0] for s in sel])
        obs = np.stack([s[1] for s in sel])
        obs = obs[:, [0, 1, 2]]
        T_gt = seq.T_rw_gt[kf.frame_idx]
        ests = [_solve_pose(_perturb(T_gt), p_w, obs, ints) for _ in range(S4["n_inits"])]
        T_est = ests[0]
        spread_m = max(np.linalg.norm(invert_se3(e)[:3, 3] - invert_se3(T_est)[:3, 3]) for e in ests)
        spread_deg = max(_rot_err_deg(e[:3, :3], T_est[:3, :3]) for e in ests)
        dt = float(np.linalg.norm(invert_se3(T_est)[:3, 3] - invert_se3(T_gt)[:3, 3]))
        dr = _rot_err_deg(T_est[:3, :3], T_gt[:3, :3])

        def cost(T):
            o, _ = rig.project(p_w @ T[:3, :3].T + T[:3, 3])
            e = np.linalg.norm(o - obs, axis=1) / sig
            return float(np.sum(0.5 * k_res**2 * np.log1p((e / k_res) ** 2)))
        c_gt, c_est = cost(T_gt), cost(T_est)
        print(f"[S4] frame {kf.frame_idx}: {len(p_w)} stereo landmarks  vs GT |dt| {1e3 * dt:.2f} mm |dR| {dr:.4f} deg  "
              f"init spread {1e3 * spread_m:.3f} mm {spread_deg:.1e} deg  Cauchy cost GT {c_gt:.2f} -> solved {c_est:.2f}")
        assert spread_m < S4["max_init_spread_m"] and spread_deg < S4["max_init_spread_deg"]
        assert c_est <= c_gt
        assert dt < S4["max_trans_err_m"] and dr < S4["max_rot_err_deg"]
        n_checked += 1
    assert n_checked >= 3
