"""OptimizationProblem with a vectorized Gauss-Newton assembly (pysteam is not modified).

pysteam's `build_gauss_newton_terms` inserts every (key1, key2) Jacobian block of every
cost term into a COO builder one at a time. With GP-interpolated observations each term
touches up to 5 states (25 blocks), and assembly dominates the run time.

Here each term's weighted, whitened Jacobian (from pysteam's own
`eval_weighted_and_whitened`) is written once into a stacked sparse J, and
A = J^T J, b = -J^T e are formed with two sparse products: the same quantities.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pysteam.problem import OptimizationProblem, WeightedLeastSquareCostTerm


class FastOptimizationProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__(use_sparse_matrix=True)

    def build_gauss_newton_terms(self):
        sv = self._state_vector
        n = sv.get_state_size()
        groups: dict[tuple[int, int], tuple[list, list, list]] = {}   # (m, d) -> (row0s, col0s, blocks)
        errs = []
        r0 = 0
        for term in self._cost_terms:
            if not isinstance(term, WeightedLeastSquareCostTerm):
                raise TypeError(f"unsupported cost term {type(term).__name__}")
            e, jacs = term.eval_weighted_and_whitened()
            m = e.shape[0]
            for key, J in jacs.items():
                if not sv.has_state_var(key):
                    continue
                c0 = sv.get_state_indices(key).start
                g = groups.setdefault(J.shape, ([], [], []))
                g[0].append(r0)
                g[1].append(c0)
                g[2].append(J)
            errs.append(e.reshape(-1))
            r0 += m
        rows, cols, vals = [], [], []
        for (m, d), (r0s, c0s, blocks) in groups.items():
            r0s = np.asarray(r0s)[:, None, None]
            c0s = np.asarray(c0s)[:, None, None]
            rows.append(np.broadcast_to(r0s + np.arange(m)[None, :, None], (len(blocks), m, d)).ravel())
            cols.append(np.broadcast_to(c0s + np.arange(d)[None, None, :], (len(blocks), m, d)).ravel())
            vals.append(np.stack(blocks).astype(np.float64).ravel())
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, n))
        e = np.concatenate(errs).reshape(-1, 1)
        return (J.T @ J).tocsr(), np.asarray(-(J.T @ e))
