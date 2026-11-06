"""M2 checks for the ORB front-end, driven by GT camera poses.

Synthetic unit tests always run. The dataset test runs when the configured sequence exists.
Run with `pytest -s` to see the mapping statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, invert_se3
from pysteam_vo.frontend.map import Map
from pysteam_vo.frontend.orb import Features, OrbExtractor, bucket_keypoints, match_descriptors
from pysteam_vo.frontend.tracker import (
    Tracker,
    parallax_deg,
    project,
    triangulate_multiview,
    triangulate_two_view,
)

CFG = load_config()
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
K = np.array([[458.0, 0.0, 367.0], [0.0, 457.0, 248.0], [0.0, 0.0, 1.0]])
RNG = np.random.default_rng(0)


def _T_cw(rotvec, c_w) -> np.ndarray:
    """Camera-from-world from a rotation and the camera center in world."""
    R_wc = Rotation.from_rotvec(rotvec).as_matrix()
    T_wc = np.eye(4)
    T_wc[:3, :3], T_wc[:3, 3] = R_wc, c_w
    return invert_se3(T_wc)


def _feats(n: int) -> Features:
    z = np.zeros((n, 2))
    return Features(z, z.copy(), np.zeros((n, 32), np.uint8), np.zeros(n, np.int64), np.zeros(n))


# ---------------------------------------------------------------- synthetic tests

def test_bucket_keypoints_caps_each_cell():
    pts = RNG.uniform([0, 0], [80, 60], size=(1000, 2))
    resp = RNG.uniform(size=1000)
    keep = bucket_keypoints(pts, resp, (60, 80), rows=3, cols=4, per_cell=5)
    cells = (pts[keep, 1] // 20).astype(int) * 4 + (pts[keep, 0] // 20).astype(int)
    assert np.bincount(cells, minlength=12).tolist() == [5] * 12
    # kept points are the strongest of their cell
    all_cells = (pts[:, 1] // 20).astype(int) * 4 + (pts[:, 0] // 20).astype(int)
    for c in range(12):
        in_c = np.flatnonzero(all_cells == c)
        np.testing.assert_array_equal(np.sort(keep[cells == c]), np.sort(in_c[np.argsort(-resp[in_c])[:5]]))


def test_extractor_spreads_features_over_grid():
    from pysteam_vo.data.euroc import CameraCalib
    calib = CameraCalib(K=K, dist=np.zeros(4), width=752, height=480, T_bc=np.eye(4))
    img = (RNG.uniform(size=(480, 752)) * 255).astype(np.uint8)
    ext = OrbExtractor(CFG["frontend"]["orb"], calib)
    f = ext.extract(img)
    o = CFG["frontend"]["orb"]
    assert 0.5 * o["nfeatures"] <= len(f) <= o["nfeatures"]
    assert f.desc.shape == (len(f), 32)
    np.testing.assert_allclose(f.kps, f.kps_dist, atol=1e-6)   # zero distortion
    r = (f.kps_dist[:, 1] * o["grid_rows"] / 480).astype(int)
    c = (f.kps_dist[:, 0] * o["grid_cols"] / 752).astype(int)
    assert len(np.unique(r * o["grid_cols"] + c)) >= 0.9 * o["grid_rows"] * o["grid_cols"]


def test_match_descriptors_recovers_permutation():
    da = RNG.integers(0, 256, size=(200, 32), dtype=np.uint8)
    perm = RNG.permutation(200)
    db = da[perm].copy()
    db[:, 0] ^= 1                                           # 1-bit noise
    pairs = match_descriptors(da, db, lowe_ratio=0.8, max_hamming=64, mutual=True)
    assert len(pairs) == 200
    np.testing.assert_array_equal(perm[pairs[:, 1]], pairs[:, 0])


def test_triangulation_exact():
    p_w = RNG.uniform([-2, -2, 4], [2, 2, 8], size=(50, 3))
    Ts = np.stack([_T_cw([0.0, 0.05 * k, 0.0], [0.3 * k, 0.05 * k, 0.0]) for k in range(4)])
    uvs = [project(K, T, p_w)[0] for T in Ts]
    np.testing.assert_allclose(triangulate_two_view(K, Ts[0], Ts[2], uvs[0], uvs[2]), p_w, atol=1e-8)
    for n in range(5):
        uv_n = np.stack([u[n] for u in uvs])
        np.testing.assert_allclose(triangulate_multiview(K, Ts, uv_n), p_w[n], atol=1e-8)
    ang = parallax_deg(np.zeros(3), np.array([1.0, 0, 0]), np.array([[0.5, 0, 0.5]]))
    np.testing.assert_allclose(ang, [90.0])


def test_map_bookkeeping_and_arrays():
    m = Map()
    k0 = m.add_keyframe(10, 100, np.eye(4), _feats(5))
    k1 = m.add_keyframe(12, 200, np.eye(4), _feats(4))
    lm = m.add_landmark(np.array([1.0, 2.0, 3.0]), np.zeros(32, np.uint8))
    m.add_observation(lm.id, k0.id, 3)
    m.add_observation(lm.id, k1.id, 1)
    with pytest.raises(ValueError):
        m.add_observation(lm.id, k1.id, 2)                  # landmark already in k1
    lm2 = m.add_landmark(np.zeros(3), np.zeros(32, np.uint8))
    with pytest.raises(ValueError):
        m.add_observation(lm2.id, k0.id, 3)                 # keypoint already used
    with pytest.raises(AssertionError):
        m.add_keyframe(13, 150, np.eye(4), _feats(1))       # out of time order
    a = m.to_arrays()
    assert a.kf_t_ns.dtype == np.int64 and a.kf_t_ns.tolist() == [100, 200]
    assert a.obs_kf.tolist() == [0, 1] and a.obs_lm.tolist() == [0, 0]
    assert m.track_lengths().tolist() == [2, 0]


# ---------------------------------------------------------------- dataset test (M2)

@pytest.fixture(scope="module")
def gt_map():
    seq = EurocSequence.from_config(CFG)
    m2 = CFG["m2_check"]
    tracker = Tracker(seq.calib, CFG["frontend"])
    stats = []
    for i in range(m2["start_frame"], m2["end_frame"]):
        if not seq.gt_valid[i]:
            continue
        T_cw = invert_se3(seq.T_wc_gt[i])
        if tracker.need_keyframe(T_cw):
            stats.append(tracker.add_keyframe(i, int(seq.image_ts_ns[i]), seq.load_image(i), T_cw))
    return tracker.map, stats, seq.calib.K


@needs_data
def test_m2_tracks_reproject_across_keyframes(gt_map):
    """M2: landmarks tracked across >= 3 GT-posed keyframes reproject under 1 px in every view."""
    m2 = CFG["m2_check"]
    the_map, stats, K_seq = gt_map
    a = the_map.to_arrays()
    lens = the_map.track_lengths()

    long_lm = np.flatnonzero(lens >= m2["min_track_len"])
    sel = np.isin(a.obs_lm, long_lm)
    errs = np.array([
        np.linalg.norm(project(K_seq, a.T_cw[k], a.p_w[l][None])[0][0] - uv)
        for k, l, uv in zip(a.obs_kf[sel], a.obs_lm[sel], a.obs_uv[sel])
    ])
    n_assoc = sum(s.n_associated for s in stats)
    n_rej = sum(s.n_assoc_rejected for s in stats)
    accept = n_assoc / max(1, n_assoc + n_rej)

    print(f"\n[M2] keyframes {len(a.kf_ids)}  landmarks {len(a.lm_ids)}  observations {len(a.obs_kf)}")
    print(f"[M2] track length: mean {lens.mean():.2f}  max {lens.max()}  "
          f">= {m2['min_track_len']}: {len(long_lm)}  hist {np.bincount(lens)[2:].tolist()}")
    print(f"[M2] associations accepted {n_assoc}  rejected {n_rej}  ratio {accept:.3f}  "
          f"retriangulated {sum(s.n_retriangulated for s in stats)}")
    print(f"[M2] long-track reproj: median {np.median(errs):.3f} px  p90 {np.percentile(errs, 90):.3f} px  "
          f"max {errs.max():.3f} px  ({len(errs)} views)")

    assert len(a.kf_ids) >= m2["min_keyframes"]
    assert len(long_lm) >= m2["min_long_tracks"]
    assert np.median(errs) < m2["max_median_reproj_px"]
    assert np.percentile(errs, 90) < m2["max_p90_reproj_px"]
    assert accept >= m2["min_assoc_accept_ratio"]

