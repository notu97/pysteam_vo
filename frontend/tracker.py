"""Keyframe mapping: association to existing landmarks and triangulation of new ones.

The keyframe pose `T_cw` is supplied by the caller. In M2 it is ground truth; in M3 it
will come from PnP against the map. Everything here is pose-source agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from pysteam_vo.data.euroc import CameraCalib
from pysteam_vo.frontend.map import Keyframe, Map
from pysteam_vo.frontend.orb import OrbExtractor, fundamental_inliers, match_descriptors


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

    def add_keyframe(self, frame_idx: int, t_ns: int, img: np.ndarray, T_cw: np.ndarray) -> KeyframeStats:
        prev_kfs = self.map.last_keyframes(int(self.c_kf["window"]))
        feats = self.extractor.extract(img)
        kf = self.map.add_keyframe(frame_idx, t_ns, T_cw, feats)
        stats = KeyframeStats(kf.id, n_features=len(feats))
        touched: set[int] = set()
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
