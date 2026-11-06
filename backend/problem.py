"""Build pysteam problems from front-end map arrays.

M5: keyframe bundle adjustment (no motion prior). One SE3StateVar (T_cw) per keyframe,
one HomoPointStateVar per landmark, one reprojection cost term per observation.

Gauge (monocular, 7 dof): the first keyframe pose is locked (6 dof) and one landmark is
locked to fix scale. The anchor is the longest-tracked landmark seen by the first keyframe,
i.e. one from the initialization pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pylgmath import Transformation
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.stereo import HomoPointStateVar
from pysteam.problem import OptimizationProblem

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

    prob = OptimizationProblem(use_sparse_matrix=True)
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
