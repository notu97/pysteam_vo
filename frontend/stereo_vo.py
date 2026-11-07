"""Stereo VO (S2): port of Orb-feat-python/vo_euroc.py onto the pysteam_vo stereo rig.

Algorithm (unchanged from the reference script):
  - each keyframe gets 3D points from row-constrained left/right ORB matching;
  - each frame is tracked against the LAST keyframe only: descriptor kNN + ratio test ->
    PnP-RANSAC (constant-velocity guess) -> LM refinement on inliers;
  - a new keyframe is made when inliers drop or motion since the keyframe is large.

Differences from the reference (intentional):
  - poses are rectified-left camera poses T_rw = T_rect,w (world = first rectified-left
    camera), not body poses;
  - a frame whose PnP fails is flagged `tracked=False`. Its pose is still the motion-model
    prediction (so the trajectory and keyframe chain continue exactly as in the reference),
    but it is excluded from coverage and accuracy figures and must not be trusted downstream.

No pysteam imports (front-end).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoRig
from pysteam_vo.frontend.orb import Features, make_extractor
from pysteam_vo.frontend.stereo import match_stereo


@dataclass
class StereoKeyframe:
    frame_idx: int
    t_ns: int
    T_rw: np.ndarray          # (4,4) rectified-left camera-from-world
    feats: Features           # left features (rectified pixels)
    pts3d: np.ndarray         # (N,3) per left feature in this camera frame, NaN if no stereo depth
    uR: np.ndarray            # (N,) right-image u of the stereo match, NaN if none


@dataclass
class StereoFrameResult:
    frame_idx: int
    t_ns: int
    T_rw: np.ndarray          # pose used (PnP result, or motion-model prediction if not tracked)
    tracked: bool             # PnP succeeded
    n_inliers: int
    is_keyframe: bool
    kf_ref: int               # index (into StereoVO.keyframes) of the keyframe tracked against


def _rt_to_T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def _T_to_rt(T: np.ndarray):
    return cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1).copy()


class StereoVO:
    def __init__(self, rig: StereoRig, cfg: dict):
        """`cfg` is the full config (uses frontend.orb, stereo, stereo_vo)."""
        self.rig = rig
        self.K = rig.K
        self.c = cfg["stereo_vo"]
        self.c_stereo = cfg["stereo"]
        self.scale_factor = float(cfg["frontend"]["orb"]["scale_factor"])
        self.extractor = make_extractor(cfg["frontend"]["orb"], rig.rect_calib)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframes: list[StereoKeyframe] = []
        self.frames: list[StereoFrameResult] = []

    # ---------------------------------------------------------------- keyframes

    def _make_keyframe(self, frame_idx: int, t_ns: int, T_rw: np.ndarray, feats: Features,
                       img_right: np.ndarray) -> StereoKeyframe:
        right = self.extractor.extract(img_right)
        sm = match_stereo(feats, right, self.c_stereo, self.K[0, 0], self.rig.baseline, self.scale_factor)
        pts3d = np.full((len(feats), 3), np.nan)
        uR = np.full(len(feats), np.nan)
        P = self.rig.triangulate(sm.uL, sm.v, sm.uR)
        keep = P[:, 2] < self.c_stereo["max_depth_m"]
        pts3d[sm.left_idx[keep]] = P[keep]
        uR[sm.left_idx[keep]] = sm.uR[keep]
        kf = StereoKeyframe(frame_idx, int(t_ns), T_rw.copy(), feats, pts3d, uR)
        self.keyframes.append(kf)
        return kf

    # ---------------------------------------------------------------- tracking

    def _track(self, feats: Features, kf: StereoKeyframe, T_ck_guess: np.ndarray):
        """Estimate T_cur,kf. Returns (T or None, n_inliers)."""
        c = self.c
        has3d = np.flatnonzero(~np.isnan(kf.pts3d[:, 0]))
        if len(has3d) < c["min_matches"] or len(feats) < c["min_matches"]:
            return None, 0
        q, t = [], []
        for m in self.matcher.knnMatch(feats.desc, kf.feats.desc[has3d], k=2):
            if len(m) < 2:
                continue
            a, b = m
            if a.distance < c["max_hamming"] and a.distance < c["lowe_ratio"] * b.distance:
                q.append(a.queryIdx)
                t.append(has3d[a.trainIdx])
        if len(q) < c["min_matches"]:
            return None, len(q)
        q, t = np.array(q), np.array(t)
        obj = kf.pts3d[t].astype(np.float64)
        img = feats.kps[q].astype(np.float64)
        rvec, tvec = _T_to_rt(T_ck_guess)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, self.K, None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            iterationsCount=int(c["pnp_ransac_iters"]), reprojectionError=float(c["pnp_reproj_px"]),
            confidence=float(c["pnp_confidence"]), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < c["min_inliers"]:
            return None, 0 if inl is None else len(inl)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        return _rt_to_T(rvec, tvec), len(inl)

    def _predict(self) -> np.ndarray:
        """Constant velocity in the world frame on the last two output poses (as the reference)."""
        T_wc_prev = invert_se3(self.frames[-1].T_rw)
        if len(self.frames) < 2:
            return self.frames[-1].T_rw
        T_wc_pp = invert_se3(self.frames[-2].T_rw)
        T_wc_pred = T_wc_prev @ (invert_se3(T_wc_pp) @ T_wc_prev)
        return invert_se3(T_wc_pred)

    # ---------------------------------------------------------------- public

    def process(self, frame_idx: int, t_ns: int, img_left: np.ndarray, img_right: np.ndarray) -> StereoFrameResult:
        """Process one rectified stereo pair."""
        feats = self.extractor.extract(img_left)
        if not self.keyframes:
            self._make_keyframe(frame_idx, t_ns, np.eye(4), feats, img_right)
            res = StereoFrameResult(frame_idx, int(t_ns), np.eye(4), True, 0, True, 0)
            self.frames.append(res)
            return res

        kf_i = len(self.keyframes) - 1
        kf = self.keyframes[kf_i]
        T_rw_pred = self._predict()
        T_ck_guess = T_rw_pred @ invert_se3(kf.T_rw)
        T_ck, n_inl = self._track(feats, kf, T_ck_guess)
        tracked = T_ck is not None
        if not tracked:
            T_ck = T_ck_guess                        # reference behaviour, but flagged
        T_rw = T_ck @ kf.T_rw

        T_kc = invert_se3(T_ck)
        rot = np.degrees(np.linalg.norm(Rotation.from_matrix(T_kc[:3, :3]).as_rotvec()))
        need_kf = (n_inl < self.c["kf_min_inliers"] or np.linalg.norm(T_kc[:3, 3]) > self.c["kf_trans_m"]
                   or rot > self.c["kf_rot_deg"])
        if need_kf:
            self._make_keyframe(frame_idx, t_ns, T_rw, feats, img_right)
        res = StereoFrameResult(frame_idx, int(t_ns), T_rw, tracked, int(n_inl), need_kf, kf_i)
        self.frames.append(res)
        return res

    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(frame_idx (F,), T_rw (F,4,4), tracked (F,) bool) for every processed frame."""
        return (np.array([f.frame_idx for f in self.frames], np.int64),
                np.stack([f.T_rw for f in self.frames]) if self.frames else np.zeros((0, 4, 4)),
                np.array([f.tracked for f in self.frames], bool))
