"""S5 checks: stereo BA problem (synthetic, fast). The EuRoC run is eval/run_s5_stereo_ba.py."""

from __future__ import annotations

import numpy as np
from pylgmath import se3op
from scipy.spatial.transform import Rotation

from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.backend.stereo_problem import build_stereo_ba_problem
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.frontend.map import MapArrays

CFG = load_config()
BE = CFG["backend"]
K = np.array([[435.2, 0.0, 367.5], [0.0, 435.2, 252.2], [0.0, 0.0, 1.0]])
B = 0.110
RNG = np.random.default_rng(55)


def _scene(n_kf=6, n_lm=250, stereo_frac=0.8, single_kf_lms=30):
    T = []
    for k in range(n_kf):
        T_wc = np.eye(4)
        T_wc[:3, :3] = Rotation.from_rotvec([0.0, 0.04 * k, 0.01 * k]).as_matrix()
        T_wc[:3, 3] = [0.3 * k, 0.03 * k, 0.02 * k]
        T.append(invert_se3(T_wc))
    T = np.stack(T)
    p_w = RNG.uniform([-2, -1.5, 3], [4, 1.5, 9], size=(n_lm + single_kf_lms, 3))
    obs_kf, obs_lm, obs_uv, obs_uR = [], [], [], []
    for l, p in enumerate(p_w):
        kfs = [int(RNG.integers(n_kf))] if l >= n_lm else range(n_kf)
        for k in kfs:
            pc = T[k, :3, :3] @ p + T[k, :3, 3]
            u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if not (0 <= u < 752 and 0 <= v < 480):
                continue
            uR = K[0, 0] * (pc[0] - B) / pc[2] + K[0, 2]
            obs_kf.append(k); obs_lm.append(l); obs_uv.append([u, v])
            obs_uR.append(uR if RNG.uniform() < stereo_frac else np.nan)
    n = len(p_w)
    arr = MapArrays(np.arange(n_kf), np.arange(n_kf), np.arange(n_kf, dtype=np.int64), T, np.arange(n), p_w,
                    np.array(obs_kf), np.array(obs_lm), np.array(obs_uv), np.array(obs_uR))
    return arr, n_lm


def test_single_keyframe_landmarks_excluded_and_factor_routing():
    arr, n_lm = _scene()
    ba = build_stereo_ba_problem(arr, K, B, BE)
    track = np.bincount(arr.obs_lm, minlength=len(arr.lm_ids))
    np.testing.assert_array_equal(ba.lm_rows, np.flatnonzero(track >= 2))
    used = np.isin(arr.obs_lm, ba.lm_rows)
    assert ba.n_obs_stereo == int(np.sum(used & np.isfinite(arr.obs_uR)))
    assert ba.n_obs_mono == int(np.sum(used & ~np.isfinite(arr.obs_uR)))
    assert ba.pose_vars[0].locked and not any(v.locked for v in ba.pose_vars[1:])
    assert not any(v.locked for v in ba.lm_vars)                       # stereo: no scale anchor


def test_stereo_ba_recovers_noise_free_geometry_with_metric_scale():
    arr, _ = _scene()
    T = arr.T_cw.copy()
    for k in range(1, len(T)):
        T[k] = se3op.vec2tran(np.r_[RNG.normal(0, 0.03, 3), RNG.normal(0, np.radians(1.0), 3)].reshape(6, 1)) @ T[k]
    p = arr.p_w * 1.05 + RNG.normal(0, 0.05, arr.p_w.shape)            # includes a 5% scale error
    pert = MapArrays(arr.kf_ids, arr.kf_frame_idx, arr.kf_t_ns, T, arr.lm_ids, p, arr.obs_kf, arr.obs_lm,
                     arr.obs_uv, arr.obs_uR)
    ba = build_stereo_ba_problem(pert, K, B, BE)
    cfg = dict(BE, relative_cost_change_threshold=1e-12, absolute_cost_change_threshold=1e-16)
    st = solve_lm(ba.problem, cfg, n_pose=ba.n_pose_dims)
    assert st.final_cost < 1e-10 * st.initial_cost
    np.testing.assert_allclose(ba.poses(), arr.T_cw, atol=1e-6)
    np.testing.assert_allclose(ba.points(), arr.p_w[ba.lm_rows], atol=1e-5)
