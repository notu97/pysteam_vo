"""Left/right ORB matching on a rectified stereo pair (ORB-SLAM-style row search).

No pysteam imports. Features come from `frontend.orb` extractors run on rectified images
(with the rig's distortion-free rectified calib), so `Features.kps` are rectified pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pysteam_vo.frontend.orb import Features


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) Hamming distances between uint8 descriptor sets (N, 32) and (M, 32)."""
    A = np.unpackbits(a, axis=1).astype(np.float32)
    B = np.unpackbits(b, axis=1).astype(np.float32)
    return A @ (1.0 - B).T + (1.0 - A) @ B.T


@dataclass(frozen=True)
class StereoMatches:
    left_idx: np.ndarray      # (M,) indices into the left Features
    right_idx: np.ndarray     # (M,) indices into the right Features
    uL: np.ndarray            # (M,) rectified pixels
    v: np.ndarray             # (M,) left row (v_L)
    uR: np.ndarray            # (M,)
    hamming: np.ndarray       # (M,)


def match_stereo(
    left: Features, right: Features, cfg: dict, fx: float, baseline: float, scale_factor: float
) -> StereoMatches:
    """Best right match for each left feature on (nearly) the same row.

    A right candidate must lie within `row_tol_px * scale_factor**octave` rows (the ORB pyramid
    scale factor), have disparity in (min_disparity_px, fx * baseline / min_depth_m), pass the
    Hamming limit and the ratio test against the second-best candidate on the row. Each right
    feature is used once. `cfg` is the `stereo` config section.
    """
    n = len(left)
    empty = np.zeros(0, np.int64)
    if n == 0 or len(right) == 0:
        z = np.zeros(0)
        return StereoMatches(empty, empty, z, z, z, z)
    xL, yL = left.kps[:, 0], left.kps[:, 1]
    xR, yR = right.kps[:, 0], right.kps[:, 1]
    D = hamming_matrix(left.desc, right.desc)
    tol = cfg["row_tol_px"] * float(scale_factor) ** left.octave.astype(np.float64)
    disp = xL[:, None] - xR[None, :]
    max_disp = fx * baseline / cfg["min_depth_m"]
    valid = (np.abs(yL[:, None] - yR[None, :]) <= tol[:, None]) & (disp > cfg["min_disparity_px"]) & (disp < max_disp)
    D[~valid] = np.inf

    best = np.argmin(D, axis=1)
    d1 = D[np.arange(n), best]
    D2 = D.copy()
    D2[np.arange(n), best] = np.inf
    d2 = D2.min(axis=1)
    ok = np.isfinite(d1) & (d1 <= cfg["max_hamming"]) & (d1 < cfg["lowe_ratio"] * d2)
    li = np.flatnonzero(ok)
    ri = best[li]
    # one-to-one: keep the best left feature per right feature
    order = np.lexsort((d1[li], ri))
    first = np.r_[True, np.diff(ri[order]) != 0]
    li, ri = li[order][first], ri[order][first]
    s = np.argsort(li)
    li, ri = li[s], ri[s]
    return StereoMatches(li, ri, xL[li], yL[li], xR[ri], d1[li])
