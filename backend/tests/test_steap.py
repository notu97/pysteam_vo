"""M6 checks: STEAP-style estimation (const-vel GP prior + non-keyframe observations).

Synthetic tests always run (seconds). The EuRoC test is marked `slow`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pylgmath import se3op
from pysteam.problem import OptimizationProblem

from pysteam_vo.backend import problem as P
from pysteam_vo.backend.cached_interp import CachedPoseInterpolator
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.frontend.map import MapArrays
from pysteam_vo.frontend.tracker import project

CFG = load_config()
BE = CFG["backend"]
M6 = CFG["m6_check"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
RNG = np.random.default_rng(6)
T0_NS = 1_403_636_600_000_000_000                              # EuRoC-sized timestamps (int64 ns)
W_TRUE = np.array([0.8, 0.05, 0.1, 0.02, 0.15, 0.03]).reshape(6, 1)   # (trans, rot) body velocity


def _T_at(t_s: float) -> np.ndarray:
    """Exact constant-velocity trajectory in the const_vel convention: T(t) = exp(t w^) T(0)."""
    return se3op.vec2tran(t_s * W_TRUE)


def _cv_scene(n_kf=6, kf_dt_ns=200_000_000, frames_per_gap=3, n_lm=250, noise_px=0.0):
    """Keyframes + in-between frames on a constant-velocity trajectory, all exact."""
    kf_t = T0_NS + np.arange(n_kf, dtype=np.int64) * kf_dt_ns
    T_kf = np.stack([_T_at((t - T0_NS) * 1e-9) for t in kf_t])
    centers = invert_se3(T_kf)[:, :3, 3]
    p_w = centers.mean(0) + RNG.uniform([-3, -2, 3], [3, 2, 8], size=(n_lm, 3))

    def visible(T):
        uv, z = project(K, T, p_w)
        return uv, (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < 752) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)

    obs_kf, obs_lm, obs_uv = [], [], []
    for k, T in enumerate(T_kf):
        uv, vis = visible(T)
        for l in np.flatnonzero(vis):
            obs_kf.append(k)
            obs_lm.append(l)
            obs_uv.append(uv[l] + (RNG.normal(0, noise_px, 2) if noise_px else 0))
    arr = MapArrays(np.arange(n_kf), np.arange(n_kf) * (frames_per_gap + 1), kf_t, T_kf, np.arange(n_lm) + 1000,
                    p_w, np.array(obs_kf), np.array(obs_lm), np.array(obs_uv))

    t_nk, lm_nk, uv_nk, T_nk = [], [], [], []
    for k in range(n_kf - 1):
        for j in range(1, frames_per_gap + 1):
            t = int(kf_t[k] + j * kf_dt_ns // (frames_per_gap + 1))
            T = _T_at((t - T0_NS) * 1e-9)
            uv, vis = visible(T)
            T_nk.append((t, T))
            for l in np.flatnonzero(vis):
                t_nk.append(t)
                lm_nk.append(l + 1000)
                uv_nk.append(uv[l] + (RNG.normal(0, noise_px, 2) if noise_px else 0))
    nonkf = (np.array(t_nk, np.int64), np.array(lm_nk, np.int64), np.array(uv_nk))
    return arr, nonkf, T_nk


def _perturb(arr: MapArrays, anchor_row: int) -> MapArrays:
    T = arr.T_cw.copy()
    for k in range(1, len(T)):
        T[k] = se3op.vec2tran(np.r_[RNG.normal(0, 0.02, 3), RNG.normal(0, np.radians(0.5), 3)].reshape(6, 1)) @ T[k]
    p = arr.p_w + RNG.normal(0, 0.03, arr.p_w.shape)
    p[anchor_row] = arr.p_w[anchor_row]
    return MapArrays(arr.kf_ids, arr.kf_frame_idx, arr.kf_t_ns, T, arr.lm_ids, p, arr.obs_kf, arr.obs_lm, arr.obs_uv)


# ---------------------------------------------------------------- building blocks

def test_velocity_init_is_exact_on_constant_velocity():
    arr, _, _ = _cv_scene()
    w = P.init_velocities_from_poses(arr.T_cw, arr.kf_t_ns)
    np.testing.assert_allclose(w, np.broadcast_to(W_TRUE, w.shape), atol=1e-9)


def test_initial_state_is_the_given_map():
    """M6 must start from the map it is given (the VO map), not from any other estimate."""
    arr, nonkf, _ = _cv_scene()
    pert = _perturb(arr, 0)
    sp = P.build_steap_problem(pert, nonkf, K, BE)
    assert np.array_equal(sp.poses(), pert.T_cw)
    np.testing.assert_array_equal(sp.points(), pert.p_w[sp.lm_rows])
    np.testing.assert_array_equal(sp.velocities(), P.init_velocities_from_poses(pert.T_cw, pert.kf_t_ns))


def test_cached_interpolator_and_fast_assembly_match_pysteam():
    """Cached interpolation + vectorized assembly give pysteam's exact cost, A and b."""
    arr, nonkf, _ = _cv_scene(noise_px=0.5)
    pert = _perturb(arr, 0)
    sp = P.build_steap_problem(pert, nonkf, K, BE)

    class Passthrough(CachedPoseInterpolator):                # same graph, no caching
        def forward(self):
            self._node = self._inner.forward()
            return self._node

        def backward(self, lhs, node, jacs):
            self._inner.backward(lhs, node, jacs)

    orig = P.CachedPoseInterpolator
    P.CachedPoseInterpolator = Passthrough
    try:
        ref = P.build_steap_problem(pert, nonkf, K, BE)
    finally:
        P.CachedPoseInterpolator = orig
    for prob in (sp.problem, ref.problem):
        prob.get_state_vector()
    A, b = sp.problem.build_gauss_newton_terms()
    A0, b0 = OptimizationProblem.build_gauss_newton_terms(ref.problem)   # pysteam's own assembly
    assert sp.problem.cost() == pytest.approx(ref.problem.cost(), rel=1e-12)
    np.testing.assert_allclose(A.toarray(), A0.toarray(), rtol=1e-9, atol=1e-9 * abs(A0).max())
    np.testing.assert_allclose(b, b0, rtol=1e-9, atol=1e-9 * np.abs(b0).max())
    # cache must follow state changes
    d = RNG.normal(0, 1e-3, (sp.problem.get_state_vector().get_state_size(), 1))
    sp.problem.get_state_vector().update(d)
    ref.problem.get_state_vector().update(d)
    assert sp.problem.cost() == pytest.approx(ref.problem.cost(), rel=1e-12)


# ---------------------------------------------------------------- constant-velocity trajectory

def test_truth_is_zero_cost_and_interpolation_exact():
    """At the true constant-velocity state the prior and all observations have zero error."""
    arr, nonkf, T_nk = _cv_scene()
    sp = P.build_steap_problem(arr, nonkf, K, BE)
    assert sp.problem.cost() < 1e-16
    for t, T in T_nk:
        np.testing.assert_allclose(sp.interpolate(np.array([t]))[0], T, atol=1e-10)


def test_steap_recovers_constant_velocity_trajectory():
    """Perturbed start, noise-free data: STEAP returns the true knots, velocities and in-between poses."""
    arr, nonkf, T_nk = _cv_scene()
    # The weak first-velocity prior is centred on the (perturbed) initial velocity, so it would
    # bias the solution slightly; make it negligible to test exact recovery.
    cfg = dict(BE, relative_cost_change_threshold=1e-12, absolute_cost_change_threshold=1e-16,
               gp=dict(BE["gp"], first_velocity_prior_sigma=1e6))
    anchor = P.build_steap_problem(arr, nonkf, K, cfg).anchor_row
    sp = P.build_steap_problem(_perturb(arr, anchor), nonkf, K, cfg)
    st = solve_lm(sp.problem, cfg, n_pose=sp.n_pose_dims)
    assert st.final_cost < 1e-10 * st.initial_cost
    np.testing.assert_allclose(sp.poses(), arr.T_cw, atol=1e-6)
    np.testing.assert_allclose(sp.velocities(), np.broadcast_to(W_TRUE, sp.velocities().shape), atol=1e-5)
    T_est = sp.interpolate(np.array([t for t, _ in T_nk]))
    np.testing.assert_allclose(T_est, np.stack([T for _, T in T_nk]), atol=1e-6)


def _accel_scene(n_kf=8, kf_dt_ns=250_000_000, frames_per_gap=4, n_lm=300, noise_px=0.5):
    """Like _cv_scene but the body velocity oscillates, so motion between knots is not constant-velocity."""
    period_s = 1.0
    def w_at(t):
        return W_TRUE * (1.0 + 0.8 * np.sin(2 * np.pi * t / period_s))
    dt_int = 1e-3
    t_end = (n_kf - 1) * kf_dt_ns * 1e-9
    ts = np.arange(0, t_end + 1e-9, dt_int)
    Ts = [np.eye(4)]
    for t in ts[:-1]:
        Ts.append(se3op.vec2tran(dt_int * w_at(t)) @ Ts[-1])
    T_of = lambda t_s: Ts[int(round(t_s / dt_int))]
    kf_t = T0_NS + np.arange(n_kf, dtype=np.int64) * kf_dt_ns
    T_kf = np.stack([T_of((t - T0_NS) * 1e-9) for t in kf_t])
    p_w = invert_se3(T_kf)[:, :3, 3].mean(0) + RNG.uniform([-3, -2, 3], [3, 2, 8], size=(n_lm, 3))

    def observe(T):
        uv, z = project(K, T, p_w)
        vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < 752) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        return [(l, uv[l] + RNG.normal(0, noise_px, 2)) for l in np.flatnonzero(vis)]

    obs = [(k, l, uv) for k, T in enumerate(T_kf) for l, uv in observe(T)]
    arr = MapArrays(np.arange(n_kf), np.arange(n_kf), kf_t, T_kf, np.arange(n_lm) + 1000, p_w,
                    np.array([o[0] for o in obs]), np.array([o[1] for o in obs]), np.array([o[2] for o in obs]))
    t_nk, lm_nk, uv_nk, T_nk = [], [], [], []
    step = kf_dt_ns // (frames_per_gap + 1)
    for k in range(n_kf - 1):
        for j in range(1, frames_per_gap + 1):
            t = int(kf_t[k] + j * step)
            T = T_of((t - T0_NS) * 1e-9)
            T_nk.append((t, T))
            for l, uv in observe(T):
                t_nk.append(t)
                lm_nk.append(l + 1000)
                uv_nk.append(uv)
    return arr, (np.array(t_nk, np.int64), np.array(lm_nk, np.int64), np.array(uv_nk)), T_nk


def test_nonkeyframe_observations_help_when_motion_is_not_constant_velocity():
    """Between knots the GP prior alone assumes constant velocity; in-between observations fix that."""
    arr, nonkf, T_nk = _accel_scene()
    anchor = P.build_steap_problem(arr, nonkf, K, BE).anchor_row
    pert = _perturb(arr, anchor)
    T_true = np.stack([T for _, T in T_nk])
    errs = {}
    for use in (False, True):
        cfg = dict(BE, gp=dict(BE["gp"], use_nonkeyframe_obs=use))
        sp = P.build_steap_problem(pert, nonkf, K, cfg)
        solve_lm(sp.problem, cfg, n_pose=sp.n_pose_dims)
        T_est = sp.interpolate(np.array([t for t, _ in T_nk]))
        c_err = invert_se3(T_est)[:, :3, 3] - invert_se3(T_true)[:, :3, 3]
        errs[use] = float(np.sqrt(np.mean(np.sum(c_err**2, axis=1))))
    print(f"\nin-between pose RMSE: knots only {1e3 * errs[False]:.2f} mm, with in-between obs {1e3 * errs[True]:.2f} mm")
    assert errs[True] < 0.7 * errs[False]


# ---------------------------------------------------------------- EuRoC (M6)

@needs_data
@pytest.mark.slow
def test_m6_steap_from_vo():
    """M6: STEAP initialized from VO improves on VO for keyframes and for every tracked frame."""
    from pysteam_vo.eval.run_m6_steap import report, run_m6

    seq = EurocSequence.from_config(CFG)
    r = run_m6(seq, CFG, M6["start_frame"], M6["end_frame"], with_ba=True)
    print()
    report(r, seq.name, M6["start_frame"], M6["end_frame"])
    assert np.isfinite(r["stats_steap"].final_cost)
    assert r["ate_kf_steap"]["rmse"] <= M6["max_ate_ratio_vs_vo"] * r["ate_kf_vo"]["rmse"]
    assert r["ate_all_steap"]["rmse"] <= M6["max_ate_ratio_vs_vo"] * r["ate_all_vo"]["rmse"]
