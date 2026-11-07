"""S1 checks: stereo rig, rectification conventions, stereo geometry on GT poses.

Synthetic tests always run. Dataset tests run when the configured sequence exists.
Run with `pytest -s` to see per-frame statistics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3, load_camera_calib
from pysteam_vo.data.stereo import StereoRig, StereoSequence
from pysteam_vo.frontend.orb import fundamental_inliers, make_extractor, match_descriptors
from pysteam_vo.frontend.stereo import match_stereo

CFG = load_config()
S1 = CFG["s1_check"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")
RNG = np.random.default_rng(11)


@pytest.fixture(scope="module")
def rig() -> StereoRig:
    if not (SEQ_DIR / "mav0").is_dir():
        pytest.skip(f"{SEQ_DIR} missing")
    m = SEQ_DIR / "mav0"
    return StereoRig.from_calibs(load_camera_calib(m / "cam0" / "sensor.yaml"), load_camera_calib(m / "cam1" / "sensor.yaml"))


# ---------------------------------------------------------------- conventions (real calibration, synthetic points)

def test_rig_basics(rig: StereoRig):
    assert 0.09 < rig.baseline < 0.13                     # EuRoC VI-sensor baseline ~0.11 m
    for R in (rig.R0, rig.R1, rig.T_B_rect[:3, :3]):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0
    assert rig.P1[1, 3] == 0 and rig.P0[0, 3] == 0      # horizontal rectification, left at origin


def test_rectified_model_matches_raw_cameras(rig: StereoRig):
    """Project points into the RAW distorted cam0/cam1 images with OpenCV, rectify those pixels,
    and compare with our rectified stereo model (u_L, v, u_R). Catches R0 direction, baseline
    sign and T_B_rect mistakes."""
    p_rect = RNG.uniform([-2, -1.5, 1.0], [2, 1.5, 15.0], size=(500, 3))
    T_B_p = rig.T_B_rect
    p_body = p_rect @ T_B_p[:3, :3].T + T_B_p[:3, 3]
    obs, z = rig.project(p_rect)
    raw = []
    for cam in (rig.cam0, rig.cam1):
        T_cb = invert_se3(cam.T_bc)
        rvec = cv2.Rodrigues(T_cb[:3, :3])[0]
        uv, _ = cv2.projectPoints(p_body, rvec, T_cb[:3, 3], cam.K, cam.dist)
        raw.append(uv.reshape(-1, 2))
    # keep points inside both raw images (distortion model is only valid there)
    w, h = rig.size
    inside = np.all([(r[:, 0] > 5) & (r[:, 0] < w - 5) & (r[:, 1] > 5) & (r[:, 1] < h - 5) for r in raw], axis=0)
    assert inside.sum() > 100
    rl = rig.rectify_points(raw[0][inside], 0)
    rr = rig.rectify_points(raw[1][inside], 1)
    np.testing.assert_allclose(rl[:, 0], obs[inside, 0], atol=1e-3)
    np.testing.assert_allclose(rl[:, 1], obs[inside, 1], atol=1e-3)
    np.testing.assert_allclose(rr[:, 0], obs[inside, 2], atol=1e-3)
    np.testing.assert_allclose(rr[:, 1], obs[inside, 1], atol=1e-3)   # same row on both sides


def test_triangulate_inverts_project(rig: StereoRig):
    p = RNG.uniform([-2, -1.5, 0.5], [2, 1.5, 30.0], size=(200, 3))
    obs, _ = rig.project(p)
    np.testing.assert_allclose(rig.triangulate(obs[:, 0], obs[:, 1], obs[:, 2]), p, rtol=1e-9, atol=1e-9)


def test_match_stereo_synthetic():
    from pysteam_vo.frontend.orb import Features
    n = 300
    desc = RNG.integers(0, 256, (n, 32), dtype=np.uint8)
    uL = RNG.uniform(100, 700, n); v = RNG.uniform(20, 460, n); d = RNG.uniform(3, 60, n)
    perm = RNG.permutation(n)
    left = Features(np.c_[uL, v], np.c_[uL, v], desc, np.zeros(n, np.int64), np.zeros(n))
    kps_r = np.c_[uL - d, v + RNG.normal(0, 0.3, n)][perm]            # same row up to 0.3 px noise
    right = Features(kps_r, kps_r.copy(), desc[perm], np.zeros(n, np.int64), np.zeros(n))
    m = match_stereo(left, right, CFG["stereo"], fx=450.0, baseline=0.11, scale_factor=1.2)
    assert len(m.left_idx) > 0.95 * n
    np.testing.assert_array_equal(perm[m.right_idx], m.left_idx)
    np.testing.assert_allclose(m.uL - m.uR, d[m.left_idx])


# ---------------------------------------------------------------- dataset (S1)

@pytest.fixture(scope="module")
def seq() -> StereoSequence:
    return StereoSequence.from_config(CFG)


def _extract(seq, i, extractor):
    l, r = seq.load_rectified(i)
    return extractor.extract(l), extractor.extract(r)


@needs_data
def test_s1_pairs_and_gt(seq: StereoSequence):
    assert len(seq) > 3000 and seq.n_unpaired == 0
    assert seq.image_ts_ns.dtype == np.int64 and np.all(np.diff(seq.image_ts_ns) > 0)
    v = seq.gt_valid
    assert 0.9 < v.mean() <= 1.0
    # rectified-left GT = cam0 GT rotated by R0^T (same optical centre)
    from pysteam_vo.data.euroc import EurocSequence
    mono = EurocSequence.from_config(CFG)
    np.testing.assert_array_equal(mono.image_ts_ns, seq.image_ts_ns)
    np.testing.assert_allclose(seq.T_w_rect_gt[v, :3, 3], mono.T_wc_gt[v, :3, 3], atol=1e-12)
    np.testing.assert_allclose(seq.T_w_rect_gt[v, :3, :3], mono.T_wc_gt[v, :3, :3] @ seq.rig.R0.T, atol=1e-12)


def _gt_pair_matches(seq, a, b, ex, cm):
    """Stereo matches at a, left matches a->b (F-RANSAC, GT-independent), joined on a's left index."""
    rig, K = seq.rig, seq.rig.K
    La, Ra = (ex.extract(x) for x in seq.load_rectified(a))
    Lb = ex.extract(seq.load_rectified(b)[0])
    sm = match_stereo(La, Ra, CFG["stereo"], K[0, 0], rig.baseline, float(CFG["frontend"]["orb"]["scale_factor"]))
    ab = match_descriptors(La.desc, Lb.desc, cm["lowe_ratio"], cm["max_hamming"], True)
    ab = ab[fundamental_inliers(La.kps[ab[:, 0]], Lb.kps[ab[:, 1]], cm["ransac_thresh_px"], cm["ransac_confidence"])]
    row_of = {int(i): k for k, i in enumerate(sm.left_idx)}
    sel = np.array([(row_of[int(i)], int(i), int(j)) for i, j in ab if int(i) in row_of], dtype=np.int64).reshape(-1, 3)
    return La, Ra, Lb, sm, sel


def _first_frame_beyond(seq, a, dist_m, max_ahead):
    c_a = seq.T_w_rect_gt[a, :3, 3]
    return next((j for j in range(a + 1, min(a + 1 + max_ahead, len(seq)))
                 if seq.gt_valid[j] and np.linalg.norm(seq.T_w_rect_gt[j, :3, 3] - c_a) >= dist_m), None)


@needs_data
def test_s1_rectified_rows(seq: StereoSequence):
    """Left/right matches found WITHOUT a row constraint lie on the same row (median < 1 px)."""
    ex = make_extractor(CFG["frontend"]["orb"], seq.rig.rect_calib)
    print()
    for a in S1["rect_anchor_frames"]:
        L, R = (ex.extract(x) for x in seq.load_rectified(a))
        pr = match_descriptors(L.desc, R.desc, S1["rect_lowe_ratio"], S1["rect_max_hamming"], True)
        row_err = np.abs(L.kps[pr[:, 0], 1] - R.kps[pr[:, 1], 1])
        print(f"[S1] frame {a}: unconstrained L/R matches {len(pr)}  row err median {np.median(row_err):.3f} px "
              f"p90 {np.percentile(row_err, 90):.2f} px")
        assert len(pr) >= S1["min_rect_matches"]
        assert np.median(row_err) < S1["max_median_row_err_px"]


@needs_data
def test_s1_metric_scale(seq: StereoSequence):
    """Stereo depth agrees with depth triangulated from two GT-posed views ~5 baselines apart."""
    from pysteam_vo.frontend.tracker import parallax_deg, triangulate_two_view
    ex = make_extractor(CFG["frontend"]["orb"], seq.rig.rect_calib)
    cm, K, T_rw = CFG["frontend"]["matching"], seq.rig.K, seq.T_rw_gt
    ratios = []
    for a in range(S1["scale_anchor_start"], S1["scale_anchor_end"], S1["scale_anchor_step"]):
        if not seq.gt_valid[a]:
            continue
        c = _first_frame_beyond(seq, a, S1["gt_baseline_m"], 300)
        if c is None:
            continue
        La, _, Lc, sm, sel = _gt_pair_matches(seq, a, c, ex, cm)
        if len(sel) == 0:
            continue
        z_st = (K[0, 0] * seq.rig.baseline / (sm.uL - sm.uR))[sel[:, 0]]
        X = triangulate_two_view(K, T_rw[a], T_rw[c], La.kps[sel[:, 1]], Lc.kps[sel[:, 2]])
        z_gt = (X @ T_rw[a][:3, :3].T + T_rw[a][:3, 3])[:, 2]
        par = parallax_deg(seq.T_w_rect_gt[a, :3, 3], seq.T_w_rect_gt[c, :3, 3], X)
        ok = (z_gt > 0) & (par >= S1["min_parallax_deg"]) & (z_st < S1["max_stereo_depth_m"])
        ratios.append(z_st[ok] / z_gt[ok])
    r = np.concatenate(ratios)
    med = float(np.median(r))
    boot = [np.median(np.random.default_rng(k).choice(r, len(r))) for k in range(500)]
    print(f"\n[S1] metric scale: {len(ratios)} anchors, {len(r)} points, median stereo/GT depth {med:.4f} "
          f"(bootstrap 95% CI {np.percentile(boot, 2.5):.4f}-{np.percentile(boot, 97.5):.4f}), baseline {seq.rig.baseline:.4f} m")
    assert len(r) >= S1["min_scale_points"]
    assert abs(med - 1.0) <= S1["max_scale_error"]


@needs_data
def test_s1_reprojection_gross_check(seq: StereoSequence):
    """Stereo points moved by GT into a later frame reproject without gross error. This is only a
    sanity check: GT noise sets a ~1.1 px floor that hides small convention/scale errors."""
    ex = make_extractor(CFG["frontend"]["orb"], seq.rig.rect_calib)
    cm, K = CFG["frontend"]["matching"], seq.rig.K
    print()
    for a in S1["rect_anchor_frames"]:
        b = _first_frame_beyond(seq, a, S1["reproj_min_baseline_m"], S1["max_frames_ahead"])
        assert b is not None
        _, _, Lb, sm, sel = _gt_pair_matches(seq, a, b, ex, cm)
        P = seq.rig.triangulate(sm.uL, sm.v, sm.uR)[sel[:, 0]]
        T_ba = seq.T_rw_gt[b] @ seq.T_w_rect_gt[a]
        Pb = P @ T_ba[:3, :3].T + T_ba[:3, 3]
        err = np.linalg.norm(Pb[:, :2] / Pb[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2] - Lb.kps[sel[:, 2]], axis=1)
        print(f"[S1] stereo pts at {a} -> frame {b}: {len(err)} pts  reproj median {np.median(err):.3f} px  "
              f"p90 {np.percentile(err, 90):.3f} px")
        assert np.median(err) < S1["max_median_reproj_px_gross"]
