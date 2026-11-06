"""ORB extraction (grid-bucketed) and descriptor matching.

Keypoints are detected on the raw (distorted) image; geometry downstream uses the
undistorted pinhole coordinates in `Features.kps`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pysteam_vo.data.euroc import CameraCalib


@dataclass(frozen=True)
class Features:
    """ORB features of one image."""

    kps_dist: np.ndarray  # (N, 2) detected pixel coords (distorted)
    kps: np.ndarray       # (N, 2) undistorted pinhole pixel coords
    desc: np.ndarray      # (N, 32) uint8 ORB descriptors
    octave: np.ndarray    # (N,) pyramid level
    response: np.ndarray  # (N,) detector response

    def __len__(self) -> int:
        return len(self.kps)


def bucket_keypoints(
    pts: np.ndarray, response: np.ndarray, shape: tuple[int, int], rows: int, cols: int, per_cell: int
) -> np.ndarray:
    """Indices of the strongest `per_cell` keypoints in each cell of a rows x cols grid."""
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    h, w = shape
    r = np.clip((pts[:, 1] * rows / h).astype(np.int64), 0, rows - 1)
    c = np.clip((pts[:, 0] * cols / w).astype(np.int64), 0, cols - 1)
    cell = r * cols + c
    order = np.lexsort((-response, cell))           # by cell, then strongest first
    cs = cell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    counts = np.diff(np.r_[starts, len(cs)])
    rank = np.arange(len(cs)) - np.repeat(starts, counts)
    return np.sort(order[rank < per_cell])


class OrbExtractor:
    """ORB detector with grid bucketing so features cover the whole image."""

    def __init__(self, cfg: dict, calib: CameraCalib):
        self.calib = calib
        self.rows = int(cfg["grid_rows"])
        self.cols = int(cfg["grid_cols"])
        self.per_cell = max(1, int(cfg["nfeatures"]) // (self.rows * self.cols))
        self._orb = cv2.ORB_create(
            nfeatures=int(cfg["detect_nfeatures"]),
            scaleFactor=float(cfg["scale_factor"]),
            nlevels=int(cfg["n_levels"]),
            edgeThreshold=int(cfg["edge_threshold"]),
            patchSize=int(cfg["edge_threshold"]),
            fastThreshold=int(cfg["fast_threshold"]),
        )

    def extract(self, img: np.ndarray) -> Features:
        kps = self._orb.detect(img, None)
        if kps:
            pts = np.array([k.pt for k in kps])
            resp = np.array([k.response for k in kps])
            keep = bucket_keypoints(pts, resp, img.shape[:2], self.rows, self.cols, self.per_cell)
            kps = [kps[i] for i in keep]
        kps, desc = self._orb.compute(img, kps)
        if desc is None or len(kps) == 0:
            z = np.zeros((0, 2))
            return Features(z, z.copy(), np.zeros((0, 32), np.uint8), np.zeros(0, np.int64), np.zeros(0))
        pts = np.array([k.pt for k in kps], dtype=np.float64)
        return Features(
            kps_dist=pts,
            kps=self.calib.undistort_points(pts),
            desc=desc,
            octave=np.array([k.octave for k in kps], dtype=np.int64),
            response=np.array([k.response for k in kps], dtype=np.float64),
        )


def match_descriptors(
    desc_a: np.ndarray, desc_b: np.ndarray, lowe_ratio: float, max_hamming: int, mutual: bool
) -> np.ndarray:
    """Match a -> b with Lowe's ratio test. Returns (M, 2) int array of (idx_a, idx_b)."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = [
        (m.queryIdx, m.trainIdx)
        for m, n in (p for p in bf.knnMatch(desc_a, desc_b, k=2) if len(p) == 2)
        if m.distance < lowe_ratio * n.distance and m.distance <= max_hamming
    ]
    out = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if mutual and len(out):
        best_ba = np.full(len(desc_b), -1, dtype=np.int64)
        for m in bf.match(desc_b, desc_a):
            best_ba[m.queryIdx] = m.trainIdx
        out = out[best_ba[out[:, 1]] == out[:, 0]]
    return out


def fundamental_inliers(pa: np.ndarray, pb: np.ndarray, thresh_px: float, confidence: float) -> np.ndarray:
    """F-matrix RANSAC inlier mask on undistorted pixel correspondences (GT-independent)."""
    n = len(pa)
    if n < 8:
        return np.zeros(n, dtype=bool)
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, thresh_px, confidence)
    if inl is None:
        return np.zeros(n, dtype=bool)
    return inl.ravel().astype(bool)
