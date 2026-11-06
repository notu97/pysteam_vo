"""M3 checks: monocular VO (essential-matrix init + projection search + PnP).

Synthetic unit tests always run. Dataset tests run when the configured sequence exists.
Run with `pytest -s` to see the per-segment statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.eval.align import ate_sim3, umeyama
from pysteam_vo.eval.plot_m3_vo import evaluate, run_vo
from pysteam_vo.frontend.orb import Features
from pysteam_vo.frontend.tracker import (
    _se3_scale,
    estimate_pose_pnp,
    initialize_two_view,
    project,
    search_by_projection,
)

CFG = load_config()
VO = CFG["frontend"]["vo"]
TRI = CFG["frontend"]["triangulation"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
K = np.array([[458.0, 0.0, 367.0], [0.0, 457.0, 248.0], [0.0, 0.0, 1.0]])
RNG = np.random.default_rng(1)


def _T_cw(rotvec, c_w) -> np.ndarray:
    T_wc = np.eye(4)
    T_wc[:3, :3], T_wc[:3, 3] = Rotation.from_rotvec(rotvec).as_matrix(), c_w
    return invert_se3(T_wc)


def _scene(n=300):
    return RNG.uniform([-3, -2, 3], [3, 2, 9], size=(n, 3))


# ---------------------------------------------------------------- synthetic tests

def test_pnp_recovers_pose_with_outliers():
    p_w = _scene()
    T_true = _T_cw([0.05, -0.1, 0.02], [0.4, -0.1, 0.2])
    uv, _ = project(K, T_true, p_w)
    uv = uv + RNG.normal(0, 0.3, uv.shape)
    out = RNG.choice(len(uv), 60, replace=False)
    uv[out] += RNG.uniform(20, 60, (60, 2))
    T_guess = _T_cw([0.0, -0.08, 0.0], [0.3, 0.0, 0.1])
    T_est, inl = estimate_pose_pnp(K, p_w, uv, T_guess, VO)
    assert T_est is not None
    assert not inl[out].any() and inl.sum() >= len(uv) - 60 - 5
    np.testing.assert_allclose(T_est, T_true, atol=5e-3)


def test_two_view_init_recovers_relative_pose_up_to_scale():
    p_w = _scene()
    T_b = _T_cw([0.0, 0.08, 0.0], [0.3, 0.02, 0.05])        # view a is the world frame
    uv_a, za = project(K, np.eye(4), p_w)
    uv_b, _ = project(K, T_b, p_w)
    out = initialize_two_view(K, uv_a + RNG.normal(0, 0.2, uv_a.shape), uv_b + RNG.normal(0, 0.2, uv_b.shape),
                              VO, TRI)
    assert out is not None
    T_est, X, good = out
    assert good.mean() > 0.9
    np.testing.assert_allclose(T_est[:3, :3], T_b[:3, :3], atol=2e-3)
    t_true = T_b[:3, 3] / np.linalg.norm(T_b[:3, 3])
    assert np.dot(T_est[:3, 3] / np.linalg.norm(T_est[:3, 3]), t_true) > 0.999
    s = VO["init_median_depth"] / np.median(za[good])            # map scale applied by init
    assert np.median(np.linalg.norm(X[good] - s * p_w[good], axis=1)) < 0.02 * s * np.median(za)


def test_two_view_init_rejects_pure_rotation():
    p_w = _scene()
    T_b = _T_cw([0.0, 0.1, 0.0], [0.0, 0.0, 0.0])
    uv_a, _ = project(K, np.eye(4), p_w)
    uv_b, _ = project(K, T_b, p_w)
    assert initialize_two_view(K, uv_a + RNG.normal(0, 0.2, uv_a.shape), uv_b, VO, TRI) is None


def test_search_by_projection_finds_true_keypoints():
    p_w = _scene(200)
    T = _T_cw([0.0, 0.05, 0.0], [0.1, 0.0, 0.0])
    uv, _ = project(K, T, p_w)
    desc = RNG.integers(0, 256, (200, 32), dtype=np.uint8)
    perm = RNG.permutation(200)
    kps = uv[perm] + RNG.normal(0, 1.0, uv.shape)
    feats = Features(kps, kps, desc[perm], np.zeros(200, np.int64), np.zeros(200))
    T_pred = _T_cw([0.0, 0.052, 0.0], [0.11, 0.0, 0.0])    # slightly wrong prediction
    pairs = search_by_projection(K, T_pred, p_w, desc, feats, (752, 480), 15.0, 64, 1.0, 0.1)
    vis = (uv[:, 0] >= 0) & (uv[:, 0] < 752) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    assert len(pairs) >= 0.95 * vis.sum()
    np.testing.assert_array_equal(perm[pairs[:, 1]], pairs[:, 0])


def test_umeyama_recovers_similarity():
    src = RNG.normal(size=(100, 3))
    R = Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix()
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    s, R_e, t = umeyama(src, dst)
    np.testing.assert_allclose([s], [2.5])
    np.testing.assert_allclose(R_e, R, atol=1e-10)
    assert ate_sim3(src, dst)["rmse"] < 1e-10


def test_se3_scale():
    T = _T_cw([0.0, 0.2, 0.0], [0.5, 0.0, 0.0])
    half = _se3_scale(T, 0.5)
    np.testing.assert_allclose(_se3_scale(T, 1.0), T, atol=1e-12)
    np.testing.assert_allclose((half @ half)[:3, :3], T[:3, :3], atol=1e-12)


# ---------------------------------------------------------------- dataset test (M3)

@needs_data
@pytest.mark.parametrize("segment", CFG["m3_check"]["segments"])
def test_m3_vo_follows_gt(segment):
    """M3: VO tracks the segment and follows GT after Sim(3) alignment."""
    m3 = CFG["m3_check"]
    seq = EurocSequence.from_config(CFG)
    start, end = segment
    vo, secs = run_vo(seq, CFG, start, end)
    r = evaluate(vo, seq, start, end)
    path = float(np.sum(np.linalg.norm(np.diff(r["c_gt"], axis=0), axis=1)))
    print(f"\n[M3] frames {start}-{end}: state={vo.state} first tracked={r['first_tracked']} "
          f"tracked {r['tracked_fraction']:.3f}  keyframes {len(vo.map.keyframes)}  "
          f"ATE rmse {r['rmse']:.3f} m over {path:.1f} m ({100 * r['rmse'] / path:.2f}%)  "
          f"scale {r['scale']:.3f}  {secs:.0f} s")
    assert vo.state == "tracking"
    assert r["tracked_fraction"] >= m3["min_tracked_fraction"]
    assert r["rmse"] <= m3["max_ate_fraction_of_path"] * path
