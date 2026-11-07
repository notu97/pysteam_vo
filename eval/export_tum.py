"""TUM trajectory I/O: `timestamp tx ty tz qx qy qz qw`, world-from-camera, seconds.

Timestamps are written from int64 nanoseconds without going through float, so they are
exact. Reading returns float seconds plus the parsed pose; use `match_to_frames` to snap
them to exact cam0 nanosecond stamps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def ns_to_str(t_ns: int) -> str:
    t_ns = int(t_ns)
    return f"{t_ns // 10**9}.{t_ns % 10**9:09d}"


def write_tum(path: str | Path, t_ns: np.ndarray, T_wc: np.ndarray) -> None:
    """Write (N,) int64 ns timestamps and (N, 4, 4) world-from-camera poses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = Rotation.from_matrix(T_wc[:, :3, :3]).as_quat()          # x, y, z, w
    with open(path, "w") as f:
        for t, T, qq in zip(t_ns, T_wc, q):
            p = T[:3, 3]
            f.write(f"{ns_to_str(t)} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n")


def read_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_s (N,) float seconds, T_wc (N, 4, 4))."""
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    a = np.array([[float(x) for x in r[:8]] for r in rows])
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def match_to_frames(t_s: np.ndarray, frame_ts_ns: np.ndarray, tol_ns: int) -> np.ndarray:
    """Index of the nearest cam0 frame for each float-seconds stamp, -1 if farther than tol_ns."""
    # split seconds to avoid float precision loss on ~1.4e9 s values
    sec = np.floor(t_s).astype(np.int64)
    frac_ns = np.round((t_s - sec) * 1e9).astype(np.int64)
    t_ns = sec * 10**9 + frac_ns
    j = np.clip(np.searchsorted(frame_ts_ns, t_ns), 1, len(frame_ts_ns) - 1)
    j = np.where(np.abs(frame_ts_ns[j - 1] - t_ns) <= np.abs(frame_ts_ns[j] - t_ns), j - 1, j)
    return np.where(np.abs(frame_ts_ns[j] - t_ns) <= tol_ns, j, -1)
