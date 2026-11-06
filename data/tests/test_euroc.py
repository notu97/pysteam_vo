"""M1 checks for the EuRoC loader.

Synthetic unit tests always run. The dataset test (GT poses + ORB-triangulated points
reproject under 1 px) runs when the configured sequence exists.
Run with `pytest -s` to see the per-pair statistics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, interpolate_poses, invert_se3

CFG = load_config()
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")


def _se3(rotvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------- synthetic tests

def test_invert_se3():
    T = _se3([0.3, -0.2, 0.9], [1.0, 2.0, -3.0])
    np.testing.assert_allclose(invert_se3(T) @ T, np.eye(4), atol=1e-12)
    Ts = np.stack([T, invert_se3(T)])
    np.testing.assert_allclose(invert_se3(Ts)[0], invert_se3(T), atol=1e-12)


def test_interpolate_constant_twist():
    """Constant-twist trajectory: interpolation at midpoints must be exact."""
    base = np.int64(1_403_636_579_763_555_584)   # realistic magnitude, int64 math
    dt = np.int64(5_000_000)
    t_src = base + dt * np.arange(5, dtype=np.int64)
    w = np.array([0.0, 0.0, 0.4])     # rad per sample, about z
    v = np.array([0.1, 0.0, 0.0])     # m per sample, in world frame
    T_src = np.stack([_se3(w * k, v * k) for k in range(5)])

    t_q = np.array([t_src[0], t_src[1] + dt // 2, t_src[-1], t_src[0] - 1, t_src[-1] + 1])
    T_q, valid = interpolate_poses(t_src, T_src, t_q, max_gap_ns=int(dt))
    np.testing.assert_array_equal(valid, [True, True, True, False, False])
    np.testing.assert_allclose(T_q[0], T_src[0], atol=1e-12)
    np.testing.assert_allclose(T_q[2], T_src[-1], atol=1e-12)
    np.testing.assert_allclose(T_q[1], _se3(w * 1.5, v * 1.5), atol=1e-12)
    assert np.all(np.isnan(T_q[3]))


def test_interpolate_rejects_gap():
    t_src = np.array([0, 5, 100], dtype=np.int64)
    T_src = np.tile(np.eye(4), (3, 1, 1))
    _, valid = interpolate_poses(t_src, T_src, np.array([3, 50], dtype=np.int64), max_gap_ns=10)
    np.testing.assert_array_equal(valid, [True, False])


# ---------------------------------------------------------------- dataset tests

@pytest.fixture(scope="module")
def seq() -> EurocSequence:
    return EurocSequence.from_config(CFG)


@needs_data
def test_loader_basics(seq: EurocSequence):
    c = seq.calib
    assert (c.width, c.height) == (752, 480)
    assert c.K[0, 0] > 0 and c.K[1, 1] > 0
    assert c.dist.shape == (4,)
    np.testing.assert_allclose(c.T_bc[:3, :3] @ c.T_bc[:3, :3].T, np.eye(3), atol=1e-6)
    assert seq.image_ts_ns.dtype == np.int64
    assert np.all(np.diff(seq.image_ts_ns) > 0)
    assert seq.image_paths[0].is_file()
    assert seq.load_image(0).shape == (c.height, c.width)

    # GT only covers part of the image stream; valid frames must lie inside GT span.
    v = seq.gt_valid
    assert 0.9 < v.mean() < 1.0
    assert seq.image_ts_ns[v].min() >= seq.gt_ts_ns[0]
    assert seq.image_ts_ns[v].max() <= seq.gt_ts_ns[-1]
    Rs = seq.T_wc_gt[v, :3, :3]
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", Rs, Rs),
                               np.broadcast_to(np.eye(3), Rs.shape), atol=1e-9)
    # Camera GT is body GT composed with T_bc at a GT sample time.
    T_wc, ok = seq.gt_cam_poses(seq.gt_ts_ns[100:101])
    assert ok[0]
    np.testing.assert_allclose(T_wc[0], seq.T_wb_gt[100] @ seq.calib.T_bc, atol=1e-12)


def _orb_matches(img_a, img_b, m):
    orb = cv2.ORB_create(nfeatures=m["orb_nfeatures"])
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < m["lowe_ratio"] * p[1].distance]
    pa = np.array([ka[g.queryIdx].pt for g in good])
    pb = np.array([kb[g.trainIdx].pt for g in good])
    return pa, pb


def _pick_second_frame(seq: EurocSequence, a: int, m) -> int | None:
    p_a = seq.T_wc_gt[a, :3, 3]
    for b in range(a + 1, min(a + 1 + m["max_frames_ahead"], len(seq))):
        if seq.gt_valid[b] and np.linalg.norm(seq.T_wc_gt[b, :3, 3] - p_a) >= m["min_baseline_m"]:
            return b
    return None


def _triangulate_and_reproject(seq: EurocSequence, a: int, b: int, m) -> dict:
    K = seq.calib.K
    pa_d, pb_d = _orb_matches(seq.load_image(a), seq.load_image(b), m)
    pa = seq.calib.undistort_points(pa_d)
    pb = seq.calib.undistort_points(pb_d)
    # Outlier rejection independent of GT (F-matrix RANSAC on undistorted pixels).
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC,
                                    m["ransac_thresh_px"], m["ransac_confidence"])
    inl = inl.ravel().astype(bool)
    pa, pb = pa[inl], pb[inl]

    T_cw_a = invert_se3(seq.T_wc_gt[a])
    T_cw_b = invert_se3(seq.T_wc_gt[b])
    X_h = cv2.triangulatePoints(K @ T_cw_a[:3], K @ T_cw_b[:3], pa.T, pb.T)
    X_w = (X_h[:3] / X_h[3]).T

    uv_a, z_a = seq.calib.project(T_cw_a, X_w)
    uv_b, z_b = seq.calib.project(T_cw_b, X_w)
    ray_a = X_w - seq.T_wc_gt[a, :3, 3]
    ray_b = X_w - seq.T_wc_gt[b, :3, 3]
    cosang = np.sum(ray_a * ray_b, 1) / (np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1))
    parallax = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    keep = (z_a > 0) & (z_b > 0) & (parallax >= m["min_parallax_deg"])

    err = np.concatenate([np.linalg.norm(uv_a - pa, axis=1)[keep],
                          np.linalg.norm(uv_b - pb, axis=1)[keep]])
    return {"a": a, "b": b, "n_match": int(len(inl)), "n_inlier": int(inl.sum()),
            "n_pts": int(keep.sum()), "median_px": float(np.median(err)) if err.size else np.inf,
            "p90_px": float(np.percentile(err, 90)) if err.size else np.inf,
            "baseline_m": float(np.linalg.norm(seq.T_wc_gt[b, :3, 3] - seq.T_wc_gt[a, :3, 3]))}


@needs_data
def test_m1_gt_triangulation_reprojects(seq: EurocSequence):
    """M1: points triangulated with GT camera poses reproject under 1 px (median)."""
    m = CFG["m1_check"]
    results = []
    for a in m["anchor_frames"]:
        assert seq.gt_valid[a], f"anchor frame {a} has no GT"
        b = _pick_second_frame(seq, a, m)
        assert b is not None, f"no frame within {m['max_frames_ahead']} of {a} reaches baseline"
        r = _triangulate_and_reproject(seq, a, b, m)
        print(f"\n[M1] frames {r['a']}->{r['b']} baseline {r['baseline_m']:.3f} m  "
              f"matches {r['n_match']} inliers {r['n_inlier']} pts {r['n_pts']}  "
              f"reproj median {r['median_px']:.3f} px  p90 {r['p90_px']:.3f} px")
        results.append(r)
    for r in results:
        assert r["n_pts"] >= m["min_points"], r
        assert r["median_px"] < m["max_median_reproj_px"], r
