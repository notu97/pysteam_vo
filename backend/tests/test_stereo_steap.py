"""S6 checks: stereo STEAP problem (synthetic, fast). The EuRoC run is eval/run_s6_stereo_steap.py."""

from __future__ import annotations

import numpy as np
from pylgmath import se3op

from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.backend.stereo_problem import build_stereo_steap_problem
from pysteam_vo.backend.tests.test_steap import K, W_TRUE, _cv_scene
from pysteam_vo.config import load_config
from pysteam_vo.frontend.map import MapArrays

CFG = load_config()
BE = CFG["backend"]
B = 0.110
RNG = np.random.default_rng(66)


def _stereo(arr: MapArrays) -> MapArrays:
    """Add right-image u for every keyframe observation of a mono synthetic scene."""
    uR = []
    for k, l in zip(arr.obs_kf, arr.obs_lm):
        pc = arr.T_cw[k, :3, :3] @ arr.p_w[l] + arr.T_cw[k, :3, 3]
        uR.append(K[0, 0] * (pc[0] - B) / pc[2] + K[0, 2])
    return MapArrays(arr.kf_ids, arr.kf_frame_idx, arr.kf_t_ns, arr.T_cw, arr.lm_ids, arr.p_w,
                     arr.obs_kf, arr.obs_lm, arr.obs_uv, np.array(uR))


def test_stereo_steap_recovers_constant_velocity_trajectory_with_scale():
    arr, nonkf, T_nk = _cv_scene()
    arr = _stereo(arr)
    T = arr.T_cw.copy()
    for k in range(1, len(T)):
        T[k] = se3op.vec2tran(np.r_[RNG.normal(0, 0.02, 3), RNG.normal(0, np.radians(0.5), 3)].reshape(6, 1)) @ T[k]
    pert = MapArrays(arr.kf_ids, arr.kf_frame_idx, arr.kf_t_ns, T, arr.lm_ids, arr.p_w * 1.04,   # 4% scale error
                     arr.obs_kf, arr.obs_lm, arr.obs_uv, arr.obs_uR)
    cfg = dict(BE, relative_cost_change_threshold=1e-12, absolute_cost_change_threshold=1e-16,
               stereo_steap=dict(BE["stereo_steap"], nonkeyframe_stride=1, first_velocity_prior_sigma=1e6))
    sp = build_stereo_steap_problem(pert, nonkf, K, B, cfg)
    assert sp.pose_vars[0].locked and not any(v.locked for v in sp.lm_vars)
    in_problem = set(arr.lm_ids[sp.lm_rows].tolist())                      # landmarks seen in >= 2 keyframes
    assert sp.n_obs_nonkf == sum(int(l) in in_problem for l in nonkf[1])
    st = solve_lm(sp.problem, cfg, n_pose=sp.n_pose_dims)
    assert st.final_cost < 1e-10 * st.initial_cost
    np.testing.assert_allclose(sp.poses(), arr.T_cw, atol=1e-6)
    np.testing.assert_allclose(sp.velocities(), np.broadcast_to(W_TRUE, sp.velocities().shape), atol=1e-5)
    np.testing.assert_allclose(sp.interpolate(np.array([t for t, _ in T_nk])), np.stack([T for _, T in T_nk]), atol=1e-6)
