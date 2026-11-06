"""Shared, cached GP pose interpolation for many observations at one timestamp.

pysteam evaluates every cost term independently, so N observations in one frame would
re-run the same interpolation graph (log, jinv, exp, compose) N times per cost evaluation
and N times per linearization. This wrapper (pysteam is not modified):

* forward: recomputes T(t) only when the values of the knot states it depends on change
  (value fingerprint, so it is always consistent with the current states);
* backward: back-propagates an identity once to get dT(t)/dx for each knot state, then
  serves `lhs @ dT/dx` to every observation. Exact, because backward is linear in lhs.
"""

from __future__ import annotations

import numpy as np
from pysteam.evaluable import Evaluable, Jacobians, Node
from pysteam.trajectory.const_vel.variable import Variable


def _state_values(knots: list[Variable]) -> np.ndarray:
    vals = []
    for kn in knots:
        vals.append(kn.pose.value.matrix().ravel())
        vals.append(np.asarray(kn.velocity.value).ravel())
    return np.concatenate(vals)


class CachedPoseInterpolator(Evaluable):
    def __init__(self, inner: Evaluable, knot1: Variable, knot2: Variable):
        super().__init__()
        self._inner = inner
        self._knots = [knot1, knot2]
        self._fwd_key: np.ndarray | None = None
        self._node: Node | None = None
        self._jac_key: np.ndarray | None = None
        self._jac: dict | None = None

    @property
    def active(self) -> bool:
        return self._inner.active

    @property
    def related_var_keys(self) -> set:
        return self._inner.related_var_keys

    def forward(self) -> Node:
        key = _state_values(self._knots)
        if self._fwd_key is None or not np.array_equal(key, self._fwd_key):
            self._node = self._inner.forward()
            self._fwd_key = key
        return self._node

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        if not self._inner.active:
            return
        if self._jac is None or not np.array_equal(self._fwd_key, self._jac_key):
            J = Jacobians()
            self._inner.backward(np.eye(6), self._node, J)
            self._jac = dict(J.items())
            self._jac_key = self._fwd_key
        for k, Jk in self._jac.items():
            jacs.add(k, lhs @ Jk)
