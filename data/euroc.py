"""EuRoC MAV (ASL format) loader.

Provides cam0 images + int64 ns timestamps, calibration (K, rad-tan distortion, T_bc)
read from sensor.yaml, and ground truth converted to *camera* poses T_wc = T_wb @ T_bc,
interpolated at image timestamps.

Pose naming: T_<to><from>. GT here is world-from-camera (T_wc); the back-end uses
T_cw. Convert with `invert_se3`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CameraCalib:
    """Pinhole + radial-tangential calibration of one EuRoC camera."""

    K: np.ndarray        # (3, 3)
    dist: np.ndarray     # (4,) k1, k2, p1, p2 (OpenCV order)
    width: int
    height: int
    T_bc: np.ndarray     # (4, 4) body-from-camera (EuRoC T_BS)

    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """Undistort (N, 2) distorted pixel coords -> (N, 2) ideal pinhole pixel coords."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2))
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def project(self, T_cw: np.ndarray, p_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) world points with T_cw into ideal (undistorted) pixels.

        Returns (uv (N, 2), depth (N,)).
        """
        p_c = (T_cw[:3, :3] @ np.asarray(p_w, dtype=np.float64).T).T + T_cw[:3, 3]
        z = p_c[:, 2]
        uv = (self.K @ (p_c / z[:, None]).T).T[:, :2]
        return uv, z


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a (4, 4) or (N, 4, 4) rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def load_camera_calib(sensor_yaml: str | Path) -> CameraCalib:
    """Parse EuRoC `camX/sensor.yaml`."""
    with open(sensor_yaml, "r") as f:
        s = yaml.safe_load(f)
    if s.get("camera_model") != "pinhole":
        raise ValueError(f"unsupported camera_model {s.get('camera_model')!r}")
    if s.get("distortion_model") != "radial-tangential":
        raise ValueError(f"unsupported distortion_model {s.get('distortion_model')!r}")
    fu, fv, cu, cv = (float(x) for x in s["intrinsics"])
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]])
    dist = np.asarray(s["distortion_coefficients"], dtype=np.float64)
    tbs = s["T_BS"]
    T_bc = np.asarray(tbs["data"], dtype=np.float64).reshape(tbs["rows"], tbs["cols"])
    width, height = (int(x) for x in s["resolution"])
    return CameraCalib(K=K, dist=dist, width=width, height=height, T_bc=T_bc)


def _read_image_list(data_csv: Path) -> tuple[np.ndarray, list[str]]:
    ts, names = [], []
    with open(data_csv, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            ts.append(int(row[0]))           # parse as int: float64 loses ns precision
            names.append(row[1].strip())
    ts_arr = np.asarray(ts, dtype=np.int64)
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], [names[i] for i in order]


def load_groundtruth(gt_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `state_groundtruth_estimate0/data.csv` -> (t_ns (M,), T_wb (M, 4, 4)).

    Quaternion columns in the file are (qw, qx, qy, qz): w first.
    """
    ts, pos, quat_wxyz = [], [], []
    with open(gt_csv, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            ts.append(int(row[0]))
            vals = [float(x) for x in row[1:8]]
            pos.append(vals[0:3])
            quat_wxyz.append(vals[3:7])
    t_ns = np.asarray(ts, dtype=np.int64)
    order = np.argsort(t_ns, kind="stable")
    t_ns = t_ns[order]
    p = np.asarray(pos)[order]
    q_wxyz = np.asarray(quat_wxyz)[order]
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]                 # scipy wants scalar last
    T_wb = np.tile(np.eye(4), (len(t_ns), 1, 1))
    T_wb[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T_wb[:, :3, 3] = p
    return t_ns, T_wb


def interpolate_poses(
    t_src_ns: np.ndarray,
    T_src: np.ndarray,
    t_query_ns: np.ndarray,
    max_gap_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate SE(3) samples (linear translation, slerp rotation) at query times.

    Returns (T_query (Q, 4, 4), valid (Q,) bool). A query is valid only if it lies
    inside [t_src[0], t_src[-1]] and its bracketing samples are <= max_gap_ns apart.
    Invalid entries are filled with NaN.
    """
    t_src_ns = np.asarray(t_src_ns, dtype=np.int64)
    t_query_ns = np.asarray(t_query_ns, dtype=np.int64)
    n_q = len(t_query_ns)
    out = np.full((n_q, 4, 4), np.nan)

    i1 = np.searchsorted(t_src_ns, t_query_ns, side="left")
    in_range = (t_query_ns >= t_src_ns[0]) & (t_query_ns <= t_src_ns[-1])
    i1 = np.clip(i1, 0, len(t_src_ns) - 1)
    exact = t_src_ns[i1] == t_query_ns
    i0 = np.where(exact, i1, np.clip(i1 - 1, 0, None))
    gap = t_src_ns[i1] - t_src_ns[i0]                     # int64, exact
    valid = in_range & (gap <= max_gap_ns)
    if not np.any(valid):
        return out, valid

    i0v, i1v = i0[valid], i1[valid]
    num = (t_query_ns[valid] - t_src_ns[i0v]).astype(np.float64)   # small ints, exact
    den = gap[valid].astype(np.float64)
    alpha = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    R0 = Rotation.from_matrix(T_src[i0v, :3, :3])
    R1 = Rotation.from_matrix(T_src[i1v, :3, :3])
    dR = (R0.inv() * R1).as_rotvec()
    R = R0 * Rotation.from_rotvec(alpha[:, None] * dR)
    p = (1.0 - alpha)[:, None] * T_src[i0v, :3, 3] + alpha[:, None] * T_src[i1v, :3, 3]

    Tv = np.tile(np.eye(4), (len(i0v), 1, 1))
    Tv[:, :3, :3] = R.as_matrix()
    Tv[:, :3, 3] = p
    out[valid] = Tv
    return out, valid


class EurocSequence:
    """One EuRoC ASL sequence, monocular (cam0 by default). No IMU / cam1 access."""

    def __init__(self, seq_dir: str | Path, camera: str, gt_max_gap_ns: int):
        self.seq_dir = Path(seq_dir)
        mav0 = self.seq_dir / "mav0"
        if not mav0.is_dir():
            raise FileNotFoundError(f"no mav0/ under {self.seq_dir}")
        cam_dir = mav0 / camera
        self.name = self.seq_dir.name
        self.calib = load_camera_calib(cam_dir / "sensor.yaml")

        self.image_ts_ns, names = _read_image_list(cam_dir / "data.csv")
        self.image_paths = [cam_dir / "data" / n for n in names]

        self.gt_ts_ns, self.T_wb_gt = load_groundtruth(
            mav0 / "state_groundtruth_estimate0" / "data.csv"
        )
        self.gt_max_gap_ns = int(gt_max_gap_ns)

        # GT camera poses at every image timestamp (NaN where GT does not cover).
        self.T_wc_gt, self.gt_valid = self.gt_cam_poses(self.image_ts_ns)

    @classmethod
    def from_config(cls, cfg: dict) -> "EurocSequence":
        d = cfg["data"]
        return cls(
            Path(d["euroc_root"]) / d["sequence"],
            camera=d["camera"],
            gt_max_gap_ns=int(d["gt_max_gap_ns"]),
        )

    def __len__(self) -> int:
        return len(self.image_ts_ns)

    def load_image(self, i: int) -> np.ndarray:
        """Grayscale uint8 image (H, W) of frame i (distorted, as recorded)."""
        img = cv2.imread(str(self.image_paths[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"failed to read {self.image_paths[i]}")
        return img

    def gt_cam_poses(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GT camera poses T_wc = T_wb @ T_bc at the given ns timestamps.

        Returns (T_wc (Q, 4, 4), valid (Q,)); invalid rows are NaN.
        """
        T_wb, valid = interpolate_poses(self.gt_ts_ns, self.T_wb_gt, t_ns, self.gt_max_gap_ns)
        T_wc = np.full_like(T_wb, np.nan)
        T_wc[valid] = T_wb[valid] @ self.calib.T_bc
        return T_wc, valid

    @property
    def T_cw_gt(self) -> np.ndarray:
        """GT camera-from-world at every image timestamp (NaN where invalid)."""
        out = np.full_like(self.T_wc_gt, np.nan)
        out[self.gt_valid] = invert_se3(self.T_wc_gt[self.gt_valid])
        return out
