"""Rectified-stereo reprojection factor for pysteam (S4).

    e = [u_L, v, u_R]_meas - g(p_c)[[0, 1, 2]],   p_c = T_rw * p_w  (4x1 homogeneous)

g is pysteam's own stereo camera model (`pysteam.evaluable.stereo.stereo_error_evaluator`):
g = [fu x/z + cu, fv y/z + cv, fu (x - w b)/z + cu, fv y/z + cv]. pysteam's StereoErrorEvaluator
uses all four rows, but on a rectified pair v_R is the same measured row as v_L, so feeding it
as an independent fourth measurement would count the vertical constraint twice. This evaluator
uses pysteam's model and Jacobian and keeps only the three real measurements.

The camera frame is the rectified left camera; intrinsics are the rectified K and baseline
(`StereoRig.K`, `StereoRig.baseline`). Pose perturbation on the left, as everywhere in pysteam.
"""

from __future__ import annotations

import numpy as np
from pysteam.evaluable import Evaluable, Jacobians, Node
from pysteam.evaluable.stereo import ComposeLandmarkEvaluator
from pysteam.evaluable.stereo.stereo_error_evaluator import CameraIntrinsics, camera_model, camera_model_jac
from pysteam.problem import CauchyLossFunc, LossFunc, StaticNoiseModel, WeightedLeastSquareCostTerm

_ROWS = [0, 1, 2]   # u_L, v_L, u_R (v_R duplicates v_L on a rectified pair)


def stereo_intrinsics(K: np.ndarray, baseline: float) -> CameraIntrinsics:
    return CameraIntrinsics(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), float(baseline))


class RectStereoErrorEvaluator(Evaluable):
    """Stereo reprojection error (u_L, v, u_R) of a camera-frame homogeneous point."""

    def __init__(self, meas: np.ndarray, intrinsics: CameraIntrinsics, point_in_cam: Evaluable):
        super().__init__()
        self._meas = np.asarray(meas, dtype=np.float64).reshape(3, 1)
        self._intrinsics = intrinsics
        self._point = point_in_cam

    @property
    def active(self) -> bool:
        return self._point.active

    @property
    def related_var_keys(self) -> set:
        return self._point.related_var_keys

    def forward(self) -> Node:
        child = self._point.forward()
        error = self._meas - camera_model(self._intrinsics, child.value)[_ROWS]
        return Node(error, child)

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        if self._point.active:
            child = node.children[0]
            J = camera_model_jac(self._intrinsics, child.value)[_ROWS]          # (3, 4)
            self._point.backward(-lhs @ J, child, jacs)


def stereo_cost_term(
    T_rw: Evaluable,
    landmark: Evaluable,
    obs: np.ndarray,
    intrinsics: CameraIntrinsics,
    sigma_px: float,
    loss: LossFunc | None = None,
    cauchy_k: float = 1.0,
) -> WeightedLeastSquareCostTerm:
    """Cost term for one stereo observation (u_L, v, u_R): isotropic pixel noise, Cauchy by default."""
    error = RectStereoErrorEvaluator(obs, intrinsics, ComposeLandmarkEvaluator(T_rw, landmark))
    noise = StaticNoiseModel(np.eye(3) * sigma_px**2, "covariance")
    return WeightedLeastSquareCostTerm(error, noise, loss if loss is not None else CauchyLossFunc(cauchy_k))
