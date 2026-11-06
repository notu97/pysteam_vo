"""Monocular pinhole reprojection factor for pysteam.

    e = uv_meas - pi(p_c),   p_c = T_cw * p_w   (4x1 homogeneous)

`uv_meas` are *undistorted* pixel coordinates (the front-end undistorts keypoints), so the
model is a plain pinhole. Modelled on pysteam's StereoErrorEvaluator: `forward` returns the
error, `backward` left-multiplies `lhs` by d(e)/d(p_c) and passes it to the child, which
is normally `ComposeLandmarkEvaluator(T_cw, landmark)` and yields the pose / landmark
Jacobians (pose perturbation on the left: T <- exp(xi^) T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pysteam.evaluable import Evaluable, Jacobians, Node
from pysteam.evaluable.stereo import ComposeLandmarkEvaluator
from pysteam.problem import CauchyLossFunc, LossFunc, StaticNoiseModel, WeightedLeastSquareCostTerm


@dataclass(frozen=True)
class PinholeIntrinsics:
    fu: float
    fv: float
    cu: float
    cv: float

    @classmethod
    def from_K(cls, K: np.ndarray) -> "PinholeIntrinsics":
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))


def pinhole_project(ints: PinholeIntrinsics, p: np.ndarray) -> np.ndarray:
    """(..., 4, 1) homogeneous camera-frame points -> (..., 2, 1) pixels (w cancels)."""
    g = np.zeros(p.shape[:-2] + (2, 1))
    g[..., 0, 0] = ints.fu * p[..., 0, 0] / p[..., 2, 0] + ints.cu
    g[..., 1, 0] = ints.fv * p[..., 1, 0] / p[..., 2, 0] + ints.cv
    return g


def pinhole_project_jac(ints: PinholeIntrinsics, p: np.ndarray) -> np.ndarray:
    """d pinhole_project / d p for homogeneous p: (..., 2, 4). Last column is zero."""
    z = p[..., 2, 0]
    J = np.zeros(p.shape[:-2] + (2, 4))
    J[..., 0, 0] = ints.fu / z
    J[..., 0, 2] = -ints.fu * p[..., 0, 0] / z**2
    J[..., 1, 1] = ints.fv / z
    J[..., 1, 2] = -ints.fv * p[..., 1, 0] / z**2
    return J


class PinholeErrorEvaluator(Evaluable):
    """Reprojection error of a camera-frame homogeneous point against a pixel measurement."""

    def __init__(self, meas: np.ndarray, intrinsics: PinholeIntrinsics, point_in_cam: Evaluable):
        super().__init__()
        self._meas = np.asarray(meas, dtype=np.float64).reshape(2, 1)
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
        error = self._meas - pinhole_project(self._intrinsics, child.value)
        return Node(error, child)

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        if self._point.active:
            child = node.children[0]
            self._point.backward(-lhs @ pinhole_project_jac(self._intrinsics, child.value), child, jacs)


def reprojection_cost_term(
    T_cw: Evaluable,
    landmark: Evaluable,
    uv: np.ndarray,
    intrinsics: PinholeIntrinsics,
    sigma_px: float,
    loss: LossFunc | None = None,
    cauchy_k: float = 1.0,
) -> WeightedLeastSquareCostTerm:
    """Cost term for one observation: isotropic pixel noise, Cauchy loss by default."""
    error = PinholeErrorEvaluator(uv, intrinsics, ComposeLandmarkEvaluator(T_cw, landmark))
    noise = StaticNoiseModel(np.eye(2) * sigma_px**2, "covariance")
    return WeightedLeastSquareCostTerm(error, noise, loss if loss is not None else CauchyLossFunc(cauchy_k))
