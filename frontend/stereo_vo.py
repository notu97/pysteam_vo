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


# ================================================================== S3: persistent stereo map

from pysteam_vo.frontend.map import Map                                     # noqa: E402
from pysteam_vo.frontend.tracker import estimate_pose_pnp, search_by_projection  # noqa: E402


@dataclass
class MapFrameResult:
    frame_idx: int
    t_ns: int
    T_rw: np.ndarray
    tracked: bool
    n_inliers: int
    is_keyframe: bool
    source: str                                   # "local_map", "last_keyframe", "gt" or "lost"
    lm_ids: np.ndarray                            # landmarks observed (PnP inliers)
    uv: np.ndarray                                # their left-image pixels (rectified)


class StereoMapVO:
    """Stereo VO with a persistent landmark map (S3).

    Each frame is matched against the landmarks of the last `local_map_keyframes` keyframes by
    projecting them with the predicted pose (search by projection), then PnP-RANSAC. If that
    fails, it falls back to S2-style descriptor matching against the last keyframe's landmarks.
    A new keyframe re-uses the landmarks it was matched to (adding an observation, with the
    right-image u when the stereo match exists) and creates landmarks only for unmatched
    stereo points. Keyframe policy and lost-frame handling are as in S2.

    `process(..., T_rw_given=...)` uses a supplied pose (e.g. GT) instead of estimating it; the
    data association is done exactly the same way, which is what the S3 association check uses.
    """

    def __init__(self, rig: StereoRig, cfg: dict):
        self.rig = rig
        self.K = rig.K
        self.c = cfg["stereo_map"]
        self.c_vo = cfg["stereo_vo"]
        self.c_stereo = cfg["stereo"]
        self.scale_factor = float(cfg["frontend"]["orb"]["scale_factor"])
        self.extractor = make_extractor(cfg["frontend"]["orb"], rig.rect_calib)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.map = Map()
        self.frames: list[MapFrameResult] = []
        self._pnp_cfg = {"pnp_min_inliers": self.c_vo["min_inliers"], "pnp_ransac_iters": self.c_vo["pnp_ransac_iters"],
                         "pnp_reproj_px": self.c_vo["pnp_reproj_px"], "pnp_confidence": self.c_vo["pnp_confidence"]}

    # ---------------------------------------------------------------- helpers

    def _predict(self) -> np.ndarray:
        T_wc_prev = invert_se3(self.frames[-1].T_rw)
        if len(self.frames) < 2:
            return self.frames[-1].T_rw
        T_wc_pp = invert_se3(self.frames[-2].T_rw)
        return invert_se3(T_wc_prev @ (invert_se3(T_wc_pp) @ T_wc_prev))

    def _local_landmarks(self) -> np.ndarray:
        ids: set[int] = set()
        for kf in self.map.last_keyframes(int(self.c["local_map_keyframes"])):
            ids.update(kf.lm_ids[kf.lm_ids >= 0].tolist())
        return np.array(sorted(ids), dtype=np.int64)

    def _lm_arrays(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ids) == 0:
            return np.zeros((0, 3)), np.zeros((0, 32), np.uint8)
        lms = [self.map.landmarks[i] for i in ids]
        return np.stack([l.p_w for l in lms]), np.stack([l.desc for l in lms])

    def _associate_local(self, feats: Features, T_rw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Search-by-projection matches (landmark ids, keypoint idx) under pose T_rw."""
        ids = self._local_landmarks()
        p_w, desc = self._lm_arrays(ids)
        wh = self.rig.size
        pairs = np.zeros((0, 2), np.int64)
        for radius in self.c["search_radius_px"]:
            pairs = search_by_projection(self.K, T_rw, p_w, desc, feats, wh, radius, self.c["search_max_hamming"],
                                         self.c["search_lowe_ratio"], self.c_stereo["min_depth_m"])
            if len(pairs) >= self.c_vo["min_inliers"]:
                break
        return ids[pairs[:, 0]] if len(pairs) else np.zeros(0, np.int64), pairs[:, 1]

    def _associate_last_kf(self, feats: Features) -> tuple[np.ndarray, np.ndarray]:
        """S2-style fallback: descriptor kNN against the last keyframe's landmark observations."""
        kf = next(reversed(self.map.keyframes.values()))
        has = np.flatnonzero(kf.lm_ids >= 0)
        if len(has) < self.c_vo["min_matches"] or len(feats) < self.c_vo["min_matches"]:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        q, t = [], []
        for m in self.matcher.knnMatch(feats.desc, kf.feats.desc[has], k=2):
            if len(m) == 2 and m[0].distance < self.c_vo["max_hamming"] and m[0].distance < self.c_vo["lowe_ratio"] * m[1].distance:
                q.append(m[0].queryIdx)
                t.append(has[m[0].trainIdx])
        return kf.lm_ids[np.array(t, np.int64)], np.array(q, np.int64)

    def _pnp(self, lm_ids: np.ndarray, kp: np.ndarray, feats: Features, T_guess: np.ndarray):
        if len(lm_ids) < self.c_vo["min_inliers"]:
            return None, np.zeros(0, bool)
        p_w, _ = self._lm_arrays(lm_ids)
        return estimate_pose_pnp(self.K, p_w, feats.kps[kp], T_guess, self._pnp_cfg)

    # ---------------------------------------------------------------- keyframes

    def _make_keyframe(self, frame_idx: int, t_ns: int, T_rw: np.ndarray, feats: Features,
                       img_right: np.ndarray, assoc_lm: np.ndarray, assoc_kp: np.ndarray) -> None:
        right = self.extractor.extract(img_right)
        sm = match_stereo(feats, right, self.c_stereo, self.K[0, 0], self.rig.baseline, self.scale_factor)
        kf = self.map.add_keyframe(frame_idx, t_ns, T_rw, feats)
        kf.uR = np.full(len(feats), np.nan)
        P = self.rig.triangulate(sm.uL, sm.v, sm.uR)
        ok = P[:, 2] < self.c_stereo["max_depth_m"]
        kf.uR[sm.left_idx[ok]] = sm.uR[ok]
        # 1) observations of existing landmarks (tracking associations)
        for lm_id, k in zip(assoc_lm, assoc_kp):
            if kf.lm_ids[k] < 0 and kf.id not in self.map.landmarks[int(lm_id)].obs:
                self.map.add_observation(int(lm_id), kf.id, int(k))
        # 2) new landmarks from unmatched stereo points
        T_wr = invert_se3(T_rw)
        for k, p in zip(sm.left_idx[ok], P[ok]):
            if kf.lm_ids[k] >= 0:
                continue
            lm = self.map.add_landmark(T_wr[:3, :3] @ p + T_wr[:3, 3], feats.desc[k])
            self.map.add_observation(lm.id, kf.id, int(k))

    def _need_keyframe(self, T_rw: np.ndarray, n_inl: int) -> bool:
        last = next(reversed(self.map.keyframes.values()))
        T_kc = last.T_cw @ invert_se3(T_rw)
        rot = np.degrees(np.linalg.norm(Rotation.from_matrix(T_kc[:3, :3]).as_rotvec()))
        return (n_inl < self.c_vo["kf_min_inliers"] or np.linalg.norm(T_kc[:3, 3]) > self.c_vo["kf_trans_m"]
                or rot > self.c_vo["kf_rot_deg"])

    # ---------------------------------------------------------------- public

    def process(self, frame_idx: int, t_ns: int, img_left: np.ndarray, img_right: np.ndarray,
                T_rw_given: np.ndarray | None = None) -> MapFrameResult:
        feats = self.extractor.extract(img_left)
        empty = np.zeros(0, np.int64)
        if not self.map.keyframes:
            T0 = np.eye(4) if T_rw_given is None else T_rw_given
            self._make_keyframe(frame_idx, t_ns, T0, feats, img_right, empty, empty)
            res = MapFrameResult(frame_idx, int(t_ns), T0, True, 0, True, "init", empty, np.zeros((0, 2)))
            self.frames.append(res)
            return res

        T_pred = self._predict() if T_rw_given is None else T_rw_given
        lm, kp = self._associate_local(feats, T_pred)
        if T_rw_given is not None:
            # supplied pose: keep associations that reproject within the PnP threshold
            p_w, _ = self._lm_arrays(lm)
            p_c = p_w @ T_rw_given[:3, :3].T + T_rw_given[:3, 3] if len(lm) else np.zeros((0, 3))
            uv = (p_c[:, :2] / p_c[:, 2:]) * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2] if len(lm) else np.zeros((0, 2))
            inl = (p_c[:, 2] > 0) & (np.linalg.norm(uv - feats.kps[kp], axis=1) <= self.c_vo["pnp_reproj_px"]) if len(lm) else np.zeros(0, bool)
            T_rw, tracked, source = T_rw_given, True, "gt"
        else:
            T_rw, inl = self._pnp(lm, kp, feats, T_pred)
            source = "local_map"
            if T_rw is None:
                lm, kp = self._associate_last_kf(feats)
                T_rw, inl = self._pnp(lm, kp, feats, T_pred)
                source = "last_keyframe"
            tracked = T_rw is not None
            if not tracked:
                T_rw, inl, source = T_pred, np.zeros(len(lm), bool), "lost"
        lm_in, kp_in = lm[inl], kp[inl]
        is_kf = self._need_keyframe(T_rw, int(inl.sum()))
        if is_kf:
            self._make_keyframe(frame_idx, t_ns, T_rw, feats, img_right, lm_in, kp_in)
        res = MapFrameResult(frame_idx, int(t_ns), T_rw, tracked, int(inl.sum()), is_kf, source, lm_in, feats.kps[kp_in])
        self.frames.append(res)
        return res

    def trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.array([f.frame_idx for f in self.frames], np.int64),
                np.stack([f.T_rw for f in self.frames]) if self.frames else np.zeros((0, 4, 4)),
                np.array([f.tracked for f in self.frames], bool))

    def nonkeyframe_observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PnP-inlier observations of tracked non-keyframe frames: (t_ns (M,), lm_id (M,), uv (M,2)).

        Left image only (the right image is not processed for non-keyframes), so these are
        monocular observations in rectified pixels.
        """
        t, lm, uv = [], [], []
        kf_frames = {kf.frame_idx for kf in self.map.keyframes.values()}
        for r in self.frames:
            if not r.tracked or r.is_keyframe or r.frame_idx in kf_frames or len(r.lm_ids) == 0:
                continue
            t.append(np.full(len(r.lm_ids), r.t_ns, np.int64))
            lm.append(r.lm_ids)
            uv.append(r.uv)
        if not t:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
        return np.concatenate(t), np.concatenate(lm).astype(np.int64), np.concatenate(uv)
