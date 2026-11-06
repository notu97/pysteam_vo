"""Keyframe mapping: association to existing landmarks and triangulation of new ones.

The keyframe pose `T_cw` is supplied by the caller. In M2 it is ground truth; in M3 it
will come from PnP against the map. Everything here is pose-source agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pysteam_vo.data.euroc import CameraCalib
from pysteam_vo.frontend.map import Keyframe, Map
from pysteam_vo.frontend.orb import Features, OrbExtractor, fundamental_inliers, match_descriptors


# ------------------------------------------------------------------ geometry helpers

def _normalized(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """(N, 2) pixels -> (N, 2) normalized image coords."""
    return (uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])


def triangulate_two_view(
    K: np.ndarray, T_cw_a: np.ndarray, T_cw_b: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray
) -> np.ndarray:
    """Linear (DLT) triangulation of N correspondences -> (N, 3) world points."""
    xa, xb = _normalized(K, uv_a), _normalized(K, uv_b)
    Pa, Pb = T_cw_a[:3], T_cw_b[:3]
    A = np.stack([
        xa[:, :1] * Pa[2] - Pa[0],
        xa[:, 1:] * Pa[2] - Pa[1],
        xb[:, :1] * Pb[2] - Pb[0],
        xb[:, 1:] * Pb[2] - Pb[1],
    ], axis=1)                                            # (N, 4, 4)
    X = np.linalg.svd(A)[2][:, -1]                        # (N, 4)
    return X[:, :3] / X[:, 3:]


def triangulate_multiview(K: np.ndarray, T_cw: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """DLT triangulation of one point from V >= 2 views -> (3,)."""
    x = _normalized(K, uv)
    P = T_cw[:, :3]                                       # (V, 3, 4)
    A = np.concatenate([x[:, :1] * P[:, 2] - P[:, 0], x[:, 1:] * P[:, 2] - P[:, 1]])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def project(K: np.ndarray, T_cw: np.ndarray, p_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) world points -> (uv (N, 2), depth (N,))."""
    p_c = p_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = p_c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (p_c[:, :2] / z[:, None]) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def parallax_deg(c_a: np.ndarray, c_b: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    ra, rb = p_w - c_a, p_w - c_b
    cos = np.sum(ra * rb, 1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ------------------------------------------------------------------ tracker

@dataclass
class KeyframeStats:
    kf_id: int
    n_features: int = 0
    n_matches: int = 0          # descriptor matches that passed F-RANSAC (summed over window)
    n_associated: int = 0       # matches to existing landmarks accepted by reprojection gate
    n_assoc_rejected: int = 0   # matches to existing landmarks rejected by the gate
    n_new_landmarks: int = 0
    n_retriangulated: int = 0


class Tracker:
    def __init__(self, calib: CameraCalib, cfg: dict):
        """`cfg` is the `frontend` section of the config."""
        self.calib = calib
        self.K = calib.K
        self.map = Map()
        self.extractor = OrbExtractor(cfg["orb"], calib)
        self.c_match = cfg["matching"]
        self.c_kf = cfg["keyframes"]
        self.c_tri = cfg["triangulation"]
        self.c_assoc = cfg["association"]

    # -------------------------------------------------------------- keyframe policy

    def need_keyframe(self, T_cw: np.ndarray) -> bool:
        """Motion-based policy: new keyframe once the camera moved/rotated enough."""
        if not self.map.keyframes:
            return True
        last = next(reversed(self.map.keyframes.values()))
        T_rel = last.T_cw @ np.linalg.inv(T_cw)            # T_{last, cur}
        trans = np.linalg.norm(T_rel[:3, 3])
        rot = np.degrees(np.linalg.norm(Rotation.from_matrix(T_rel[:3, :3]).as_rotvec()))
        return trans >= self.c_kf["min_translation_m"] or rot >= self.c_kf["min_rotation_deg"]

    # -------------------------------------------------------------- mapping

    def add_keyframe(
        self,
        frame_idx: int,
        t_ns: int,
        img: np.ndarray | None,
        T_cw: np.ndarray,
        feats: Features | None = None,
        assoc: np.ndarray | None = None,
    ) -> KeyframeStats:
        """Insert a keyframe with pose T_cw, associate it to the map and triangulate new points.

        `feats` skips re-extraction when the frame was already processed (VO tracking);
        `assoc` is an optional (M, 2) array of (keypoint idx, landmark id) already
        established for this frame (e.g. PnP inliers), added before descriptor matching.
        """
        prev_kfs = self.map.last_keyframes(int(self.c_kf["window"]))
        if feats is None:
            feats = self.extractor.extract(img)
        kf = self.map.add_keyframe(frame_idx, t_ns, T_cw, feats)
        stats = KeyframeStats(kf.id, n_features=len(feats))
        touched: set[int] = set()
        if assoc is not None:
            for i, lm_id in assoc:
                if kf.lm_ids[i] < 0 and kf.id not in self.map.landmarks[lm_id].obs:
                    self.map.add_observation(int(lm_id), kf.id, int(i))
                    touched.add(int(lm_id))
                    stats.n_associated += 1
        for prev in prev_kfs:                                # oldest first: widest baseline
            self._match_against(kf, prev, stats, touched)
        self._retriangulate(touched, stats)
        return stats

    def _match_against(self, kf: Keyframe, prev: Keyframe, stats: KeyframeStats, touched: set[int]) -> None:
        cm = self.c_match
        pairs = match_descriptors(kf.feats.desc, prev.feats.desc, cm["lowe_ratio"], cm["max_hamming"], cm["mutual"])
        if len(pairs) < cm["min_matches"]:
            return
        inl = fundamental_inliers(kf.feats.kps[pairs[:, 0]], prev.feats.kps[pairs[:, 1]],
                                  cm["ransac_thresh_px"], cm["ransac_confidence"])
        pairs = pairs[inl]
        stats.n_matches += len(pairs)

        # 1) Matches whose previous keypoint already has a landmark -> associate (gated).
        lm_prev = prev.lm_ids[pairs[:, 1]]
        gate = self.c_assoc["max_reproj_px"]
        for (i, _), lm_id in zip(pairs[lm_prev >= 0], lm_prev[lm_prev >= 0]):
            lm = self.map.landmarks[lm_id]
            if kf.lm_ids[i] >= 0 or kf.id in lm.obs:
                continue
            uv, z = project(self.K, kf.T_cw, lm.p_w[None])
            if z[0] > 0 and np.linalg.norm(uv[0] - kf.feats.kps[i]) <= gate:
                self.map.add_observation(lm_id, kf.id, i)
                touched.add(int(lm_id))
                stats.n_associated += 1
            else:
                stats.n_assoc_rejected += 1

        # 2) Matches with no landmark on either side -> triangulate a new landmark.
        cand = pairs[(prev.lm_ids[pairs[:, 1]] < 0) & (kf.lm_ids[pairs[:, 0]] < 0)]
        if len(cand) == 0:
            return
        ct = self.c_tri
        uv_n, uv_p = kf.feats.kps[cand[:, 0]], prev.feats.kps[cand[:, 1]]
        X = triangulate_two_view(self.K, kf.T_cw, prev.T_cw, uv_n, uv_p)
        pn, zn = project(self.K, kf.T_cw, X)
        pp, zp = project(self.K, prev.T_cw, X)
        ok = (
            (zn >= ct["min_depth_m"]) & (zp >= ct["min_depth_m"])
            & (zn <= ct["max_depth_m"]) & (zp <= ct["max_depth_m"])
            & (parallax_deg(kf.center_w, prev.center_w, X) >= ct["min_parallax_deg"])
            & (np.linalg.norm(pn - uv_n, axis=1) <= ct["max_reproj_px"])
            & (np.linalg.norm(pp - uv_p, axis=1) <= ct["max_reproj_px"])
        )
        for (i, j), p_w in zip(cand[ok], X[ok]):
            lm = self.map.add_landmark(p_w, prev.feats.desc[j])
            self.map.add_observation(lm.id, prev.id, j)
            self.map.add_observation(lm.id, kf.id, i)
            stats.n_new_landmarks += 1

    def _retriangulate(self, lm_ids: set[int], stats: KeyframeStats) -> None:
        """Refine landmarks seen in >= N views with multi-view DLT; keep it only if all views agree."""
        min_obs = int(self.c_tri["retriangulate_min_obs"])
        gate = self.c_assoc["max_reproj_px"]
        for lm_id in lm_ids:
            lm = self.map.landmarks[lm_id]
            if len(lm.obs) < min_obs:
                continue
            T, uv = self.map.observations(lm_id)
            p = triangulate_multiview(self.K, T, uv)
            errs, ok = [], True
            for Ti, uvi in zip(T, uv):
                pr, z = project(self.K, Ti, p[None])
                ok &= bool(z[0] >= self.c_tri["min_depth_m"])
                errs.append(np.linalg.norm(pr[0] - uvi))
            if ok and max(errs) <= gate:
                lm.p_w = p
                stats.n_retriangulated += 1


# ================================================================== monocular VO (M3)

_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def hamming(desc: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Hamming distances between rows of (N, 32) `desc` and one (32,) descriptor."""
    return _POPCOUNT[np.bitwise_xor(desc, d)].sum(1)


def search_by_projection(
    K: np.ndarray,
    T_cw: np.ndarray,
    p_w: np.ndarray,
    lm_desc: np.ndarray,
    feats: Features,
    image_wh: tuple[int, int],
    radius_px: float,
    max_hamming: int,
    lowe_ratio: float,
    min_depth: float,
) -> np.ndarray:
    """Match landmarks to keypoints near their projection under a predicted pose.

    Returns (M, 2) int array of (landmark row in p_w, keypoint idx); each keypoint used once.
    """
    if len(p_w) == 0 or len(feats) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    uv, z = project(K, T_cw, p_w)
    w, h = image_wh
    vis = np.flatnonzero((z >= min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    if len(vis) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    tree = cKDTree(feats.kps)
    best: dict[int, tuple[int, int]] = {}                 # kp -> (distance, landmark row)
    for li, cands in zip(vis, tree.query_ball_point(uv[vis], r=radius_px)):
        if not cands:
            continue
        c = np.asarray(cands)
        d = hamming(feats.desc[c], lm_desc[li])
        o = np.argsort(d)
        d0 = int(d[o[0]])
        if d0 > max_hamming or (len(o) > 1 and d0 > lowe_ratio * d[o[1]]):
            continue
        k = int(c[o[0]])
        if k not in best or d0 < best[k][0]:
            best[k] = (d0, int(li))
    return np.array([(li, k) for k, (_, li) in best.items()], dtype=np.int64).reshape(-1, 2)


def _se3_scale(T: np.ndarray, a: float) -> np.ndarray:
    """Approximate T^a for a small rigid motion (rotation via rotvec, translation linearly)."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(a * Rotation.from_matrix(T[:3, :3]).as_rotvec()).as_matrix()
    out[:3, 3] = a * T[:3, 3]
    return out


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def _T_to_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1).copy()


def estimate_pose_pnp(
    K: np.ndarray, p_w: np.ndarray, uv: np.ndarray, T_cw_guess: np.ndarray | None, cfg: dict
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC PnP + LM refinement on undistorted pixels. Returns (T_cw or None, inlier mask)."""
    n = len(p_w)
    mask = np.zeros(n, dtype=bool)
    if n < cfg["pnp_min_inliers"]:
        return None, mask
    obj, img = np.ascontiguousarray(p_w, np.float64), np.ascontiguousarray(uv, np.float64)
    guess = T_cw_guess is not None
    rvec, tvec = _T_to_rt(T_cw_guess) if guess else (np.zeros((3, 1)), np.zeros((3, 1)))
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, None, rvec, tvec, useExtrinsicGuess=guess,
        iterationsCount=int(cfg["pnp_ransac_iters"]), reprojectionError=float(cfg["pnp_reproj_px"]),
        confidence=float(cfg["pnp_confidence"]), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inl is None or len(inl) < cfg["pnp_min_inliers"]:
        return None, mask
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    T_cw = _rt_to_T(rvec, tvec)
    uv_p, z = project(K, T_cw, obj)
    mask = (z > 0) & (np.linalg.norm(uv_p - img, axis=1) <= cfg["pnp_reproj_px"])
    if mask.sum() < cfg["pnp_min_inliers"]:
        return None, np.zeros(n, dtype=bool)
    return T_cw, mask


def initialize_two_view(
    K: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray, cfg: dict, tri_cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Essential-matrix initialization between two views (a = world frame).

    Returns (T_cw_b, p_w (N, 3), good (N,)) with scale fixed so that the median depth of
    the good points in view a equals cfg["init_median_depth"], or None if degenerate.
    """
    if len(uv_a) < cfg["init_min_points"]:
        return None
    E, mask = cv2.findEssentialMat(uv_a, uv_b, K, cv2.RANSAC, cfg["init_ransac_confidence"],
                                   cfg["init_ransac_thresh_px"])
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, mask = cv2.recoverPose(E, uv_a, uv_b, K, mask=mask.copy())
    inl = mask.ravel() > 0
    T_cw_b = np.eye(4)
    T_cw_b[:3, :3], T_cw_b[:3, 3] = R, t.ravel()
    T_cw_a = np.eye(4)
    X = triangulate_two_view(K, T_cw_a, T_cw_b, uv_a, uv_b)
    pa, za = project(K, T_cw_a, X)
    pb, zb = project(K, T_cw_b, X)
    par = parallax_deg(np.zeros(3), -R.T @ t.ravel(), X)
    good = (
        inl & (za > 0) & (zb > 0) & (par >= tri_cfg["min_parallax_deg"])
        & (np.linalg.norm(pa - uv_a, axis=1) <= tri_cfg["max_reproj_px"])
        & (np.linalg.norm(pb - uv_b, axis=1) <= tri_cfg["max_reproj_px"])
    )
    if good.sum() < cfg["init_min_points"] or np.median(par[good]) < cfg["init_min_median_parallax_deg"]:
        return None
    s = cfg["init_median_depth"] / np.median(za[good])
    T_cw_b[:3, 3] *= s
    return T_cw_b, X * s, good


@dataclass
class FrameResult:
    frame_idx: int
    t_ns: int
    T_cw: np.ndarray | None          # None = not tracked
    n_inliers: int = 0
    is_keyframe: bool = False
    lm_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # tracked landmarks
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))           # their undistorted pixels


class MonoVO:
    """Monocular VO: essential-matrix init, then projection search + PnP per frame.

    Map scale is arbitrary (fixed at init). World frame = camera frame of the first keyframe.
    """

    def __init__(self, calib: CameraCalib, cfg: dict):
        """`cfg` is the `frontend` section of the config (needs a `vo` subsection)."""
        self.calib = calib
        self.K = calib.K
        self.c = cfg["vo"]
        self.tracker = Tracker(calib, cfg)
        self.map = self.tracker.map
        self.state = "init"                                 # init -> tracking -> lost
        self.frames: dict[int, FrameResult] = {}
        self._ref: tuple[int, int, Features] | None = None  # init reference (frame_idx, t_ns, feats)

    # ---------------------------------------------------------------- public

    def process(self, frame_idx: int, t_ns: int, img: np.ndarray) -> FrameResult:
        feats = self.tracker.extractor.extract(img)
        if self.state == "init":
            res = self._initialize(frame_idx, t_ns, feats)
        elif self.state == "tracking":
            res = self._track(frame_idx, t_ns, feats)
        else:
            res = FrameResult(frame_idx, t_ns, None)
        self.frames[frame_idx] = res
        return res

    def tracked_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """(frame_idx (F,), T_cw (F, 4, 4)) of every tracked frame, in frame order."""
        items = sorted((i, r.T_cw) for i, r in self.frames.items() if r.T_cw is not None)
        if not items:
            return np.zeros(0, np.int64), np.zeros((0, 4, 4))
        return np.array([i for i, _ in items], np.int64), np.stack([T for _, T in items])

    def nonkeyframe_observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PnP-inlier observations of tracked frames that are not keyframes.

        Returns (t_ns (M,) int64, lm_id (M,), uv (M, 2) undistorted px). Keyframe
        observations are already in the map and are not repeated here.
        """
        t, lm, uv = [], [], []
        kf_frames = {kf.frame_idx for kf in self.map.keyframes.values()}
        for i in sorted(self.frames):
            r = self.frames[i]
            if r.T_cw is None or r.is_keyframe or i in kf_frames or len(r.lm_ids) == 0:
                continue
            t.append(np.full(len(r.lm_ids), r.t_ns, dtype=np.int64))
            lm.append(r.lm_ids)
            uv.append(r.uv)
        if not t:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
        return np.concatenate(t), np.concatenate(lm).astype(np.int64), np.concatenate(uv)

    # ---------------------------------------------------------------- init

    def _initialize(self, frame_idx: int, t_ns: int, feats: Features) -> FrameResult:
        c = self.c
        if self._ref is None or frame_idx - self._ref[0] > c["init_max_frames"]:
            self._ref = (frame_idx, t_ns, feats)
            return FrameResult(frame_idx, t_ns, None)
        ref_idx, ref_t, ref_f = self._ref
        cm = self.tracker.c_match
        pairs = match_descriptors(ref_f.desc, feats.desc, cm["lowe_ratio"], cm["max_hamming"], cm["mutual"])
        if len(pairs) < c["init_min_points"]:
            return FrameResult(frame_idx, t_ns, None)
        uv_a, uv_b = ref_f.kps[pairs[:, 0]], feats.kps[pairs[:, 1]]
        if np.median(np.linalg.norm(uv_b - uv_a, axis=1)) < c["init_min_flow_px"]:
            return FrameResult(frame_idx, t_ns, None)
        out = initialize_two_view(self.K, uv_a, uv_b, c, self.tracker.c_tri)
        if out is None:
            return FrameResult(frame_idx, t_ns, None)
        T_cw_b, X, good = out

        kf_a = self.map.add_keyframe(ref_idx, ref_t, np.eye(4), ref_f)
        kf_b = self.map.add_keyframe(frame_idx, t_ns, T_cw_b, feats)
        for (i, j), p in zip(pairs[good], X[good]):
            lm = self.map.add_landmark(p, ref_f.desc[i])
            self.map.add_observation(lm.id, kf_a.id, int(i))
            self.map.add_observation(lm.id, kf_b.id, int(j))
        self.frames[ref_idx] = FrameResult(ref_idx, ref_t, np.eye(4), int(good.sum()), True)
        self.state = "tracking"
        lm_ids = kf_b.lm_ids[kf_b.lm_ids >= 0]
        return FrameResult(frame_idx, t_ns, T_cw_b, int(good.sum()), True,
                           lm_ids, feats.kps[kf_b.lm_ids >= 0])

    # ---------------------------------------------------------------- tracking

    def _predict(self, frame_idx: int) -> np.ndarray:
        """Constant-velocity motion model on the last two tracked poses, scaled by frame gaps.

        The last two tracked frames need not be adjacent (e.g. right after init), so the
        relative motion is converted to a per-frame rate before extrapolating.
        """
        tracked = [r for r in sorted(self.frames.values(), key=lambda r: r.frame_idx) if r.T_cw is not None]
        last = tracked[-1]
        if len(tracked) < 2:
            return last.T_cw
        prev = tracked[-2]
        rate = (frame_idx - last.frame_idx) / (last.frame_idx - prev.frame_idx)
        return _se3_scale(last.T_cw @ np.linalg.inv(prev.T_cw), rate) @ last.T_cw

    def _local_landmarks(self) -> np.ndarray:
        ids: set[int] = set()
        for kf in self.map.last_keyframes(int(self.c["local_map_keyframes"])):
            ids.update(kf.lm_ids[kf.lm_ids >= 0].tolist())
        return np.array(sorted(ids), dtype=np.int64)

    def _track(self, frame_idx: int, t_ns: int, feats: Features) -> FrameResult:
        c = self.c
        T_pred = self._predict(frame_idx)
        lm_ids = self._local_landmarks()
        p_w = np.stack([self.map.landmarks[i].p_w for i in lm_ids]) if len(lm_ids) else np.zeros((0, 3))
        desc = np.stack([self.map.landmarks[i].desc for i in lm_ids]) if len(lm_ids) else np.zeros((0, 32), np.uint8)
        wh = (self.calib.width, self.calib.height)
        min_depth = self.tracker.c_tri["min_depth_m"]

        T_cw, inl, pairs = None, np.zeros(0, bool), np.zeros((0, 2), np.int64)
        for radius in c["search_radius_px"]:                # narrow first, widen on failure
            pairs = search_by_projection(self.K, T_pred, p_w, desc, feats, wh, radius,
                                         c["search_max_hamming"], c["search_lowe_ratio"], min_depth)
            T_cw, inl = estimate_pose_pnp(self.K, p_w[pairs[:, 0]], feats.kps[pairs[:, 1]], T_pred, c)
            if T_cw is not None:
                break
        if T_cw is None:
            self.state = "lost"
            return FrameResult(frame_idx, t_ns, None)

        pairs = pairs[inl]
        trk_lm, trk_kp = lm_ids[pairs[:, 0]], pairs[:, 1]
        res = FrameResult(frame_idx, t_ns, T_cw, len(pairs), False, trk_lm, feats.kps[trk_kp])
        if self._need_keyframe(frame_idx, len(pairs)):
            self.tracker.add_keyframe(frame_idx, t_ns, None, T_cw, feats=feats,
                                      assoc=np.stack([trk_kp, trk_lm], axis=1))
            res.is_keyframe = True
        return res

    def _need_keyframe(self, frame_idx: int, n_tracked: int) -> bool:
        """Scale-free policy: tracked landmarks dropped vs. the last keyframe, or too long since one."""
        c = self.c
        last = next(reversed(self.map.keyframes.values()))
        gap = frame_idx - last.frame_idx
        if gap < c["kf_min_frame_gap"]:
            return False
        # Reference: landmarks tracked when the last keyframe was made (not those it then triangulated).
        last_res = self.frames.get(last.frame_idx)
        ref = last_res.n_inliers if last_res is not None else int((last.lm_ids >= 0).sum())
        return (n_tracked < c["kf_tracked_ratio"] * ref or n_tracked < c["kf_min_tracked"]
                or gap >= c["kf_max_frame_gap"])
