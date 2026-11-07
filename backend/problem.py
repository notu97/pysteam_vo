"""Build pysteam problems from front-end map arrays.

M5: keyframe bundle adjustment (no motion prior). One SE3StateVar (T_cw) per keyframe,
one HomoPointStateVar per landmark, one reprojection cost term per observation.

M6: STEAP-style estimation. Keyframes become knots of a constant-velocity GP trajectory
(pose T_cw + body velocity w), with GP prior factors between consecutive knots. Keyframe
observations act on knot poses; non-keyframe observations act on the GP-interpolated pose
at their timestamp, so they constrain both bracketing knots. Same landmarks and gauge as M5.

Gauge (monocular, 7 dof): the first keyframe pose is locked (6 dof) and one landmark is
locked to fix scale. The anchor is the longest-tracked landmark seen by the first keyframe,
i.e. one from the initialization pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pylgmath import Transformation, se3op
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.stereo import HomoPointStateVar
from pysteam.evaluable.vspace import VSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.trajectory import Time
from pysteam.trajectory import const_vel

from pysteam_vo.backend.cached_interp import CachedPoseInterpolator
from pysteam_vo.backend.fast_problem import FastOptimizationProblem
from pysteam_vo.backend.pinhole_error import PinholeIntrinsics, reprojection_cost_term
from pysteam_vo.frontend.map import MapArrays


@dataclass
class BAProblem:
    problem: OptimizationProblem
    pose_vars: list[SE3StateVar]       # one per keyframe, same order as MapArrays.kf_*
    lm_vars: list[HomoPointStateVar]   # one per used landmark
    lm_rows: np.ndarray                # (L',) rows of MapArrays.lm_* that are in the problem
    anchor_row: int                    # MapArrays row of the scale-anchor landmark
    n_obs: int

    @property
    def n_pose_dims(self) -> int:
        """Active pose dimensions; poses are added before landmarks, so they come first."""
        return 6 * sum(not v.locked for v in self.pose_vars)

    def poses(self) -> np.ndarray:
        """(K, 4, 4) current T_cw."""
        return np.stack([v.value.matrix() for v in self.pose_vars])

    def points(self) -> np.ndarray:
        """(L', 3) current landmark positions (homogeneous w is always 1 here)."""
        return np.stack([v.value[:3, 0] / v.value[3, 0] for v in self.lm_vars])


def select_scale_anchor(arr: MapArrays, track_len: np.ndarray, usable: np.ndarray) -> int:
    """Longest-tracked usable landmark observed by the first keyframe."""
    in_kf0 = np.zeros(len(arr.lm_ids), dtype=bool)
    in_kf0[arr.obs_lm[arr.obs_kf == 0]] = True
    cand = np.flatnonzero(in_kf0 & usable)
    if len(cand) == 0:
        raise ValueError("no usable landmark observed by the first keyframe to anchor scale")
    return int(cand[np.argmax(track_len[cand])])


def build_ba_problem(arr: MapArrays, K: np.ndarray, cfg: dict) -> BAProblem:
    """`cfg` is the `backend` config section."""
    ints = PinholeIntrinsics.from_K(K)
    ba = cfg["ba"]
    track_len = np.bincount(arr.obs_lm, minlength=len(arr.lm_ids))
    usable = track_len >= ba["min_track_len"]
    anchor = select_scale_anchor(arr, track_len, usable)

    pose_vars = [SE3StateVar(Transformation(T_ba=T.copy()), locked=(k == 0)) for k, T in enumerate(arr.T_cw)]
    lm_rows = np.flatnonzero(usable)
    row_to_var = {}
    lm_vars = []
    for r in lm_rows:
        v = HomoPointStateVar(np.r_[arr.p_w[r], 1.0].reshape(4, 1), locked=(r == anchor))
        row_to_var[int(r)] = v
        lm_vars.append(v)

    prob = FastOptimizationProblem()
    prob.add_state_var(*pose_vars, *lm_vars)
    n_obs = 0
    for k, l, uv in zip(arr.obs_kf, arr.obs_lm, arr.obs_uv):
        lv = row_to_var.get(int(l))
        if lv is None:
            continue
        prob.add_cost_term(reprojection_cost_term(pose_vars[k], lv, uv, ints, cfg["pixel_sigma_px"],
                                                  cauchy_k=cfg["cauchy_k"]))
        n_obs += 1
    return BAProblem(prob, pose_vars, lm_vars, lm_rows, anchor, n_obs)


# ================================================================== M6: GP trajectory


def init_velocities_from_poses(T_cw: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Knot velocities w_k = log(T_{k+1} T_k^-1) / dt (finite differences of the initial poses).

    Matches the const_vel prior, whose pose error is log(T2 T1^-1) - w1 dt. The last knot
    reuses the previous interval. Returns (K, 6, 1) in pylgmath order (translation, rotation).
    """
    n = len(T_cw)
    w = np.zeros((n, 6, 1))
    for k in range(n - 1):
        dt = (int(t_ns[k + 1]) - int(t_ns[k])) * 1e-9
        w[k] = se3op.tran2vec(T_cw[k + 1] @ np.linalg.inv(T_cw[k])) / dt
    if n > 1:
        w[-1] = w[-2]
    return w


@dataclass
class SteapProblem:
    problem: OptimizationProblem
    traj: const_vel.Interface
    pose_vars: list[SE3StateVar]
    vel_vars: list[VSpaceStateVar]
    lm_vars: list[HomoPointStateVar]
    lm_rows: np.ndarray
    anchor_row: int
    knot_t_ns: np.ndarray
    n_obs_kf: int
    n_obs_nonkf: int
    n_prior: int

    @property
    def n_pose_dims(self) -> int:
        """Active knot dims (pose + velocity); knots are added before landmarks."""
        return 6 * sum(not v.locked for v in self.pose_vars) + 6 * sum(not v.locked for v in self.vel_vars)

    def poses(self) -> np.ndarray:
        return np.stack([v.value.matrix() for v in self.pose_vars])

    def velocities(self) -> np.ndarray:
        return np.stack([v.value.copy() for v in self.vel_vars])

    def points(self) -> np.ndarray:
        return np.stack([v.value[:3, 0] / v.value[3, 0] for v in self.lm_vars])

    def interpolate(self, t_ns: np.ndarray) -> np.ndarray:
        """(Q, 4, 4) T_cw at query times from the GP (inside the knot span)."""
        return np.stack([self.traj.get_pose_interpolator(Time(nsecs=int(t))).evaluate().matrix() for t in t_ns])


def build_steap_problem(
    arr: MapArrays,
    nonkf: tuple[np.ndarray, np.ndarray, np.ndarray],
    K: np.ndarray,
    cfg: dict,
    landmark_rows: np.ndarray | None = None,
    anchor_row: int | None = None,
) -> SteapProblem:
    """`arr`: VO keyframe map (initial values). `nonkf`: (t_ns, lm_id, uv) non-keyframe observations.

    `cfg` is the `backend` config section (uses `ba` and `gp`). `landmark_rows` / `anchor_row`
    force the landmark set and scale anchor (rows of `arr.lm_*`), so two problems built from
    different keyframe subsets share the same landmarks and gauge (used by M7).
    """
    ints = PinholeIntrinsics.from_K(K)
    gp = cfg["gp"]
    track_len = np.bincount(arr.obs_lm, minlength=len(arr.lm_ids))
    if landmark_rows is None:
        usable = track_len >= cfg["ba"]["min_track_len"]
    else:
        usable = np.zeros(len(arr.lm_ids), dtype=bool)
        usable[np.asarray(landmark_rows)] = True
    anchor = select_scale_anchor(arr, track_len, usable) if anchor_row is None else int(anchor_row)
    assert usable[anchor], "scale anchor must be one of the landmarks"

    # knots: initial poses straight from VO, velocities by finite differences of those poses
    w0 = init_velocities_from_poses(arr.T_cw, arr.kf_t_ns)
    traj = const_vel.Interface(qcd=np.asarray(gp["qcd"], dtype=np.float64))
    pose_vars, vel_vars = [], []
    for k, (T, t) in enumerate(zip(arr.T_cw, arr.kf_t_ns)):
        pv = SE3StateVar(Transformation(T_ba=T.copy()), locked=(k == 0))
        vv = VSpaceStateVar(w0[k].copy())
        traj.add_knot(Time(nsecs=int(t)), pv, vv)
        pose_vars.append(pv)
        vel_vars.append(vv)
    traj.add_velocity_prior(Time(nsecs=int(arr.kf_t_ns[0])), w0[0].copy(),
                            np.eye(6) * float(gp["first_velocity_prior_sigma"]) ** 2)

    lm_rows = np.flatnonzero(usable)
    row_of_id = {int(i): r for r, i in enumerate(arr.lm_ids)}
    row_to_var = {}
    lm_vars = []
    for r in lm_rows:
        v = HomoPointStateVar(np.r_[arr.p_w[r], 1.0].reshape(4, 1), locked=(r == anchor))
        row_to_var[int(r)] = v
        lm_vars.append(v)

    prob = FastOptimizationProblem()
    for pv, vv in zip(pose_vars, vel_vars):                  # knot states first (Schur ordering)
        prob.add_state_var(pv, vv)
    prob.add_state_var(*lm_vars)

    sigma, ck = cfg["pixel_sigma_px"], cfg["cauchy_k"]
    n_kf = 0
    for k, l, uv in zip(arr.obs_kf, arr.obs_lm, arr.obs_uv):
        lv = row_to_var.get(int(l))
        if lv is not None:
            prob.add_cost_term(reprojection_cost_term(pose_vars[k], lv, uv, ints, sigma, cauchy_k=ck))
            n_kf += 1

    n_nonkf = 0
    t_obs, lm_obs, uv_obs = nonkf
    if gp["use_nonkeyframe_obs"] and len(t_obs):
        t_first, t_last = int(arr.kf_t_ns[0]), int(arr.kf_t_ns[-1])
        frame_times = np.unique(t_obs)
        frame_times = frame_times[(frame_times > t_first) & (frame_times < t_last)]
        frame_times = frame_times[:: int(gp["nonkeyframe_stride"])]
        keep = np.isin(t_obs, frame_times)
        knot_vars = [const_vel.Variable(Time(nsecs=int(t)), pv, vv)
                     for t, pv, vv in zip(arr.kf_t_ns, pose_vars, vel_vars)]
        interp_cache: dict[int, object] = {}
        for t, lid, uv in zip(t_obs[keep], lm_obs[keep], uv_obs[keep]):
            r = row_of_id.get(int(lid))
            lv = row_to_var.get(r) if r is not None else None
            if lv is None:
                continue
            T_t = interp_cache.get(int(t))
            if T_t is None:                                  # one shared, cached interpolator per frame
                k = int(np.searchsorted(arr.kf_t_ns, t)) - 1    # t strictly inside (t_k, t_k+1)
                k1, k2 = knot_vars[k], knot_vars[k + 1]
                T_t = interp_cache[int(t)] = CachedPoseInterpolator(
                    const_vel.PoseInterpolator(Time(nsecs=int(t)), k1, k2), k1, k2)
            prob.add_cost_term(reprojection_cost_term(T_t, lv, uv, ints, sigma, cauchy_k=ck))
            n_nonkf += 1

    priors = traj.get_prior_cost_terms()
    prob.add_cost_term(*priors)
    return SteapProblem(prob, traj, pose_vars, vel_vars, lm_vars, lm_rows, anchor,
                        arr.kf_t_ns.copy(), n_kf, n_nonkf, len(priors))
