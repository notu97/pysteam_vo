"""EuRoC stereo rig (cam0 + cam1): rectification, rectified camera model, GT poses.

Conventions (see CLAUDE.md):
- The stereo camera frame is the *rectified left camera* (`rect`), not raw cam0.
  cv2.stereoRectify rotates cam0 by R0 (x_rect = R0 x_cam0), so T_C0_rect has rotation R0^T
  and T_B_rect = T_B_C0 @ T_C0_rect.
- Rectified images share K = P0[:3, :3]; the right camera sits at +baseline along x_rect.
- A stereo observation is (u_L, v, u_R) in rectified pixels (v_R == v_L).
- Knots / poses are camera-from-world, here T_rect,w ("T_rw").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pysteam_vo.data.euroc import (
    CameraCalib,
    _read_image_list,
    interpolate_poses,
    invert_se3,
    load_camera_calib,
    load_groundtruth,
)


@dataclass
class StereoRig:
    cam0: CameraCalib
    cam1: CameraCalib
    R0: np.ndarray            # (3,3) rectifying rotation of cam0: x_rect = R0 x_cam0
    R1: np.ndarray            # (3,3) rectifying rotation of cam1
    P0: np.ndarray            # (3,4) rectified left projection
    P1: np.ndarray            # (3,4) rectified right projection
    K: np.ndarray             # (3,3) rectified intrinsics (shared by both rectified images)
    baseline: float           # metres, > 0
    T_B_rect: np.ndarray      # (4,4) body-from-rectified-left
    maps0: tuple[np.ndarray, np.ndarray]
    maps1: tuple[np.ndarray, np.ndarray]

    @property
    def size(self) -> tuple[int, int]:
        return self.cam0.width, self.cam0.height

    @property
    def rect_calib(self) -> CameraCalib:
        """Distortion-free calib of the rectified left image (for extractors / projection)."""
        return CameraCalib(K=self.K.copy(), dist=np.zeros(4), width=self.cam0.width,
                           height=self.cam0.height, T_bc=self.T_B_rect.copy())

    @classmethod
    def from_calibs(cls, cam0: CameraCalib, cam1: CameraCalib) -> "StereoRig":
        T_C1_C0 = invert_se3(cam1.T_bc) @ cam0.T_bc          # cam0 frame -> cam1 frame
        size = (cam0.width, cam0.height)
        R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
            cam0.K, cam0.dist, cam1.K, cam1.dist, size, T_C1_C0[:3, :3], T_C1_C0[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        maps0 = cv2.initUndistortRectifyMap(cam0.K, cam0.dist, R0, P0, size, cv2.CV_32FC1)
        maps1 = cv2.initUndistortRectifyMap(cam1.K, cam1.dist, R1, P1, size, cv2.CV_32FC1)
        K = P0[:3, :3].copy()
        baseline = float(-P1[0, 3] / P1[0, 0])
        T_C0_rect = np.eye(4)
        T_C0_rect[:3, :3] = R0.T
        return cls(cam0, cam1, R0, R1, P0, P1, K, baseline, cam0.T_bc @ T_C0_rect, maps0, maps1)

    # ------------------------------------------------------------------ images

    def rectify(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (cv2.remap(img0, *self.maps0, cv2.INTER_LINEAR),
                cv2.remap(img1, *self.maps1, cv2.INTER_LINEAR))

    def rectify_points(self, pts_raw: np.ndarray, cam: int) -> np.ndarray:
        """Raw (distorted) pixel coords of cam0/cam1 -> rectified pixel coords, (N,2)."""
        c, R, P = (self.cam0, self.R0, self.P0) if cam == 0 else (self.cam1, self.R1, self.P1)
        # cv2.undistortPoints' default 5 iterations leave up to ~0.45 px error near the image
        # edges with EuRoC's k1 ~ -0.28; iterate to convergence instead.
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        out = cv2.undistortPointsIter(np.asarray(pts_raw, np.float64).reshape(-1, 1, 2), c.K, c.dist, R, P, crit)
        return out.reshape(-1, 2)

    # ------------------------------------------------------------------ stereo model

    def project(self, p_rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) points in the rectified-left frame -> ((N,3) [u_L, v, u_R], (N,) depth)."""
        p = np.asarray(p_rect, np.float64)
        z = p[:, 2]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uL = fx * p[:, 0] / z + cx
            v = fy * p[:, 1] / z + cy
            uR = fx * (p[:, 0] - self.baseline) / z + cx
        return np.stack([uL, v, uR], axis=1), z

    def triangulate(self, uL: np.ndarray, v: np.ndarray, uR: np.ndarray) -> np.ndarray:
        """Rectified stereo observations -> (N,3) points in the rectified-left frame."""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        disp = np.asarray(uL, np.float64) - np.asarray(uR, np.float64)
        z = fx * self.baseline / disp
        return np.stack([(uL - cx) * z / fx, (v - cy) * z / fy, z], axis=1)


class StereoSequence:
    """One EuRoC sequence as rectified stereo pairs. No IMU access."""

    def __init__(self, seq_dir: str | Path, gt_max_gap_ns: int):
        self.seq_dir = Path(seq_dir)
        mav0 = self.seq_dir / "mav0"
        if not mav0.is_dir():
            raise FileNotFoundError(f"no mav0/ under {self.seq_dir}")
        self.name = self.seq_dir.name
        self.rig = StereoRig.from_calibs(load_camera_calib(mav0 / "cam0" / "sensor.yaml"),
                                         load_camera_calib(mav0 / "cam1" / "sensor.yaml"))
        t0, n0 = _read_image_list(mav0 / "cam0" / "data.csv")
        t1, n1 = _read_image_list(mav0 / "cam1" / "data.csv")
        common, i0, i1 = np.intersect1d(t0, t1, assume_unique=True, return_indices=True)
        self.image_ts_ns = common.astype(np.int64)
        self.paths0 = [mav0 / "cam0" / "data" / n0[i] for i in i0]
        self.paths1 = [mav0 / "cam1" / "data" / n1[i] for i in i1]
        self.n_unpaired = int(len(t0) + len(t1) - 2 * len(common))

        self.gt_ts_ns, self.T_wb_gt = load_groundtruth(mav0 / "state_groundtruth_estimate0" / "data.csv")
        self.gt_max_gap_ns = int(gt_max_gap_ns)
        T_wb, self.gt_valid = interpolate_poses(self.gt_ts_ns, self.T_wb_gt, self.image_ts_ns, self.gt_max_gap_ns)
        self.T_w_rect_gt = np.full_like(T_wb, np.nan)
        self.T_w_rect_gt[self.gt_valid] = T_wb[self.gt_valid] @ self.rig.T_B_rect

    @classmethod
    def from_config(cls, cfg: dict) -> "StereoSequence":
        d = cfg["data"]
        return cls(Path(d["euroc_root"]) / d["sequence"], gt_max_gap_ns=int(d["gt_max_gap_ns"]))

    def __len__(self) -> int:
        return len(self.image_ts_ns)

    def load_raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        imgs = [cv2.imread(str(p[i]), cv2.IMREAD_GRAYSCALE) for p in (self.paths0, self.paths1)]
        for im, p in zip(imgs, (self.paths0[i], self.paths1[i])):
            if im is None:
                raise IOError(f"failed to read {p}")
        return imgs[0], imgs[1]

    def load_rectified(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.rig.rectify(*self.load_raw(i))

    @property
    def T_rw_gt(self) -> np.ndarray:
        """GT rectified-left camera-from-world at every pair (NaN where GT is missing)."""
        out = np.full_like(self.T_w_rect_gt, np.nan)
        out[self.gt_valid] = invert_se3(self.T_w_rect_gt[self.gt_valid])
        return out
