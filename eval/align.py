"""Trajectory alignment and ATE (same definition as `evo_ape -as` on positions)."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ~= s * R @ src + t. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_sim3(p_est: np.ndarray, p_gt: np.ndarray) -> dict:
    """Sim(3)-align estimated positions to GT; return scale, transform, aligned positions and ATE."""
    s, R, t = umeyama(p_est, p_gt, with_scale=True)
    aligned = s * p_est @ R.T + t
    err = np.linalg.norm(aligned - p_gt, axis=1)
    return {"scale": s, "R": R, "t": t, "aligned": aligned, "err": err,
            "rmse": float(np.sqrt(np.mean(err ** 2))), "median": float(np.median(err)), "max": float(err.max())}


def rpe_translation(T_wc_est: np.ndarray, T_wc_gt: np.ndarray, step: int) -> np.ndarray:
    """Alignment-free relative translation error over windows of `step` consecutive poses
    (window starts every step/10 poses, so windows overlap). Returns per-window errors in the
    units of the poses (metres for stereo)."""
    errs = []
    stride = max(1, step // 10)
    for i in range(0, len(T_wc_est) - step, stride):
        A = np.linalg.inv(T_wc_est[i]) @ T_wc_est[i + step]
        B = np.linalg.inv(T_wc_gt[i]) @ T_wc_gt[i + step]
        errs.append(np.linalg.norm((np.linalg.inv(B) @ A)[:3, 3]))
    return np.array(errs)
