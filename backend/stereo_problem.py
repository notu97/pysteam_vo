"""Stereo bundle adjustment problem (S5) from a stereo MapArrays (rectified-left camera frame).

States: one SE3StateVar (T_rw) per keyframe and one HomoPointStateVar per landmark seen in
>= `stereo_ba.min_track_kf` keyframes. A landmark seen by a single keyframe carries no
information about the poses (3 unknowns vs at most 3 stereo measurements), so it is omitted.

Factors: observations with a right-image match (finite obs_uR) get the rectified stereo factor
(u_L, v, u_R); the others get the monocular pinhole factor on the rectified K. Cauchy loss.

Gauge: only the first keyframe pose is locked. Scale is observed through the stereo baseline.
State order is all poses, then landmarks, as `SchurLevMarqSolver` requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pylgmath import Transformation
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.stereo import HomoPointStateVar

from pysteam_vo.backend.fast_problem import FastOptimizationProblem
from pysteam_vo.backend.pinhole_error import PinholeIntrinsics, reprojection_cost_term
from pysteam_vo.backend.stereo_error import stereo_cost_term, stereo_intrinsics
from pysteam_vo.frontend.map import MapArrays


@dataclass
class StereoBAProblem:
    problem: FastOptimizationProblem
    pose_vars: list[SE3StateVar]
    lm_vars: list[HomoPointStateVar]
    lm_rows: np.ndarray
    n_obs_stereo: int
    n_obs_mono: int

    @property
    def n_pose_dims(self) -> int:
        return 6 * sum(not v.locked for v in self.pose_vars)

    def poses(self) -> np.ndarray:
        return np.stack([v.value.matrix() for v in self.pose_vars])

    def points(self) -> np.ndarray:
        return np.stack([v.value[:3, 0] / v.value[3, 0] for v in self.lm_vars])


def build_stereo_ba_problem(arr: MapArrays, K: np.ndarray, baseline: float, cfg: dict) -> StereoBAProblem:
    """`cfg` is the `backend` config section (uses `stereo_ba`)."""
    if arr.obs_uR is None:
        raise ValueError("MapArrays has no stereo observations (obs_uR)")
    sb = cfg["stereo_ba"]
    pin = PinholeIntrinsics.from_K(K)
    ste = stereo_intrinsics(K, baseline)
    track_kf = np.bincount(arr.obs_lm, minlength=len(arr.lm_ids))
    lm_rows = np.flatnonzero(track_kf >= sb["min_track_kf"])

    pose_vars = [SE3StateVar(Transformation(T_ba=T.copy()), locked=(k == 0)) for k, T in enumerate(arr.T_cw)]
    row_to_var = {}
    lm_vars = []
    for r in lm_rows:
        v = HomoPointStateVar(np.r_[arr.p_w[r], 1.0].reshape(4, 1))
        row_to_var[int(r)] = v
        lm_vars.append(v)

    prob = FastOptimizationProblem()
    prob.add_state_var(*pose_vars, *lm_vars)
    sigma, ck = cfg["pixel_sigma_px"], cfg["cauchy_k"]
    n_st = n_mo = 0
    for k, l, uv, uR in zip(arr.obs_kf, arr.obs_lm, arr.obs_uv, arr.obs_uR):
        lv = row_to_var.get(int(l))
        if lv is None:
            continue
        if np.isfinite(uR):
            prob.add_cost_term(stereo_cost_term(pose_vars[k], lv, np.r_[uv, uR], ste, sigma, cauchy_k=ck))
            n_st += 1
        else:
            prob.add_cost_term(reprojection_cost_term(pose_vars[k], lv, uv, pin, sigma, cauchy_k=ck))
            n_mo += 1
    return StereoBAProblem(prob, pose_vars, lm_vars, lm_rows, n_st, n_mo)


# ================================================================== S6: stereo STEAP

from pysteam.evaluable.vspace import VSpaceStateVar                         # noqa: E402
from pysteam.trajectory import Time, const_vel                              # noqa: E402

from pysteam_vo.backend.cached_interp import CachedPoseInterpolator         # noqa: E402
from pysteam_vo.backend.problem import SteapProblem, init_velocities_from_poses  # noqa: E402


def build_stereo_steap_problem(
    arr: MapArrays,
    nonkf: tuple[np.ndarray, np.ndarray, np.ndarray],
    K: np.ndarray,
    baseline: float,
    cfg: dict,
) -> SteapProblem:
    """Stereo STEAP: S5's landmarks and keyframe factors + const-vel GP knots + non-keyframe
    (monocular, left image) observations through the cached GP interpolator.

    Knot poses and landmarks come from `arr` (the VO map); velocities are finite differences
    of those poses. Gauge: first knot pose locked (stereo observes scale). `cfg` is the
    `backend` section (uses `stereo_ba` and `stereo_steap`).
    """
    sb, ss = cfg["stereo_ba"], cfg["stereo_steap"]
    pin = PinholeIntrinsics.from_K(K)
    ste = stereo_intrinsics(K, baseline)
    track_kf = np.bincount(arr.obs_lm, minlength=len(arr.lm_ids))
    lm_rows = np.flatnonzero(track_kf >= sb["min_track_kf"])

    w0 = init_velocities_from_poses(arr.T_cw, arr.kf_t_ns)
    traj = const_vel.Interface(qcd=np.asarray(ss["qcd"], dtype=np.float64))
    pose_vars, vel_vars, knots = [], [], []
    for k, (T, t) in enumerate(zip(arr.T_cw, arr.kf_t_ns)):
        pv = SE3StateVar(Transformation(T_ba=T.copy()), locked=(k == 0))
        vv = VSpaceStateVar(w0[k].copy())
        traj.add_knot(Time(nsecs=int(t)), pv, vv)
        knots.append(const_vel.Variable(Time(nsecs=int(t)), pv, vv))
        pose_vars.append(pv)
        vel_vars.append(vv)
    traj.add_velocity_prior(Time(nsecs=int(arr.kf_t_ns[0])), w0[0].copy(),
                            np.eye(6) * float(ss["first_velocity_prior_sigma"]) ** 2)

    row_of_id = {int(i): r for r, i in enumerate(arr.lm_ids)}
    row_to_var, lm_vars = {}, []
    for r in lm_rows:
        v = HomoPointStateVar(np.r_[arr.p_w[r], 1.0].reshape(4, 1))
        row_to_var[int(r)] = v
        lm_vars.append(v)

    prob = FastOptimizationProblem()
    for pv, vv in zip(pose_vars, vel_vars):                  # knot states first (Schur ordering)
        prob.add_state_var(pv, vv)
    prob.add_state_var(*lm_vars)
    sigma, ck = cfg["pixel_sigma_px"], cfg["cauchy_k"]

    n_kf = 0
    for k, l, uv, uR in zip(arr.obs_kf, arr.obs_lm, arr.obs_uv, arr.obs_uR):
        lv = row_to_var.get(int(l))
        if lv is None:
            continue
        if np.isfinite(uR):
            prob.add_cost_term(stereo_cost_term(pose_vars[k], lv, np.r_[uv, uR], ste, sigma, cauchy_k=ck))
        else:
            prob.add_cost_term(reprojection_cost_term(pose_vars[k], lv, uv, pin, sigma, cauchy_k=ck))
        n_kf += 1

    n_nonkf = 0
    t_obs, lm_obs, uv_obs = nonkf
    if ss["use_nonkeyframe_obs"] and len(t_obs):
        frame_t = np.unique(t_obs)
        frame_t = frame_t[(frame_t > arr.kf_t_ns[0]) & (frame_t < arr.kf_t_ns[-1])][:: int(ss["nonkeyframe_stride"])]
        keep = np.isin(t_obs, frame_t)
        cache: dict[int, CachedPoseInterpolator] = {}
        for t, lid, uv in zip(t_obs[keep], lm_obs[keep], uv_obs[keep]):
            r = row_of_id.get(int(lid))
            lv = row_to_var.get(r) if r is not None else None
            if lv is None:
                continue
            T_t = cache.get(int(t))
            if T_t is None:
                k = int(np.searchsorted(arr.kf_t_ns, t)) - 1
                k1, k2 = knots[k], knots[k + 1]
                T_t = cache[int(t)] = CachedPoseInterpolator(
                    const_vel.PoseInterpolator(Time(nsecs=int(t)), k1, k2), k1, k2)
            prob.add_cost_term(reprojection_cost_term(T_t, lv, uv, pin, sigma, cauchy_k=ck))
            n_nonkf += 1

    priors = traj.get_prior_cost_terms()
    prob.add_cost_term(*priors)
    return SteapProblem(prob, traj, pose_vars, vel_vars, lm_vars, lm_rows, -1, arr.kf_t_ns.copy(),
                        n_kf, n_nonkf, len(priors))
