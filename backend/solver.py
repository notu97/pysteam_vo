"""Batch Levenberg-Marquardt driver for pysteam problems."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.linalg as npla
import scipy.linalg as sla
import scipy.sparse as sp
from pysteam.problem import OptimizationProblem
from pysteam.solver import LevMarqGaussNewtonSolver


class SchurLevMarqSolver(LevMarqGaussNewtonSolver):
    """pysteam LM with a Schur-complement linear solve for bundle-adjustment structure.

    Requires the active state ordering [poses (n_pose dims) | landmarks (3 dims each)],
    which holds when pose variables are added to the problem before landmark variables.
    Landmark blocks of the Hessian are 3x3 block-diagonal, so they are eliminated exactly
    and only the dense (n_pose x n_pose) reduced system is factorized. Mathematically the
    same step as the plain sparse solve, just much faster. pysteam itself is unchanged:
    only `solve_gauss_newton` (also used inside LM) is overridden.
    """

    def __init__(self, problem: OptimizationProblem, n_pose: int, **parameters) -> None:
        self._n_pose = int(n_pose)
        super().__init__(problem, **parameters)

    def solve_gauss_newton(self, A, b: np.ndarray) -> np.ndarray:
        A = sp.csr_matrix(A)
        n = self._n_pose
        n_l = (A.shape[0] - n) // 3
        assert n + 3 * n_l == A.shape[0], "state is not [poses | 3-dim landmarks]"
        B = A[:n, :n].toarray()
        E = A[:n, n:]
        C = A[n:, n:].tocoo()
        if np.any(C.row // 3 != C.col // 3):
            raise ValueError("landmark block of the Hessian is not 3x3 block-diagonal")
        Cb = np.zeros((n_l, 3, 3))
        np.add.at(Cb, (C.row // 3, C.row % 3, C.col % 3), C.data)
        try:
            Cinv = npla.inv(Cb)
        except npla.LinAlgError as e:
            raise npla.LinAlgError("singular landmark block") from e
        Cinv = sp.bsr_matrix((Cinv, np.arange(n_l), np.arange(n_l + 1)), shape=(3 * n_l, 3 * n_l)).tocsr()
        ECinv = E @ Cinv
        S = B - (ECinv @ E.T).toarray()
        rhs = b[:n] - ECinv @ b[n:]
        try:
            dx_p = sla.cho_solve(sla.cho_factor(S), rhs)
        except sla.LinAlgError as e:
            raise npla.LinAlgError("reduced camera system not positive definite") from e
        dx_l = Cinv @ (b[n:] - E.T @ dx_p)
        return np.vstack([dx_p, dx_l])


@dataclass
class SolveStats:
    initial_cost: float
    final_cost: float
    iterations: int
    termination: str
    seconds: float


def solve_lm(problem: OptimizationProblem, cfg: dict, verbose: bool = False, n_pose: int | None = None) -> SolveStats:
    """Run LM to convergence. `cfg` is the `backend` config section.

    With `n_pose` (active pose dims, ordered first) the Schur-complement solver is used.

    pysteam raises RuntimeError when no LM step reduces the cost; at a minimum that is
    convergence, so it is reported as a termination cause rather than propagated. A
    non-finite cost (e.g. a singular system from an unfixed gauge) raises.
    """
    c0 = problem.cost()
    t0 = time.perf_counter()
    params = dict(
        verbose=verbose,
        max_iterations=int(cfg["max_iterations"]),
        relative_cost_change_threshold=float(cfg["relative_cost_change_threshold"]),
        absolute_cost_change_threshold=float(cfg["absolute_cost_change_threshold"]),
    )
    if n_pose is None:
        solver = LevMarqGaussNewtonSolver(problem, **params)
    else:
        solver = SchurLevMarqSolver(problem, n_pose, **params)
    try:
        solver.optimize()
        term = solver.termination_cause
    except RuntimeError as e:
        term = f"STOPPED: {e}"
    c1 = problem.cost()
    if not np.isfinite(c1):
        raise FloatingPointError(f"non-finite cost after LM ({term}); is the gauge fixed?")
    return SolveStats(float(c0), float(c1), solver.curr_iteration, term, time.perf_counter() - t0)
