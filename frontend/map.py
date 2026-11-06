"""Keyframes, landmarks and observation tracks.

Plain NumPy bookkeeping; `Map.to_arrays()` is the hand-off format for the back-end.
Poses are camera-from-world `T_cw`; landmark positions are world-frame `p_w`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pysteam_vo.frontend.orb import Features


@dataclass
class Keyframe:
    id: int
    frame_idx: int
    t_ns: int
    T_cw: np.ndarray        # (4, 4)
    feats: Features
    lm_ids: np.ndarray      # (N,) landmark id per keypoint, -1 if none

    @property
    def center_w(self) -> np.ndarray:
        """Camera center in world frame."""
        R, t = self.T_cw[:3, :3], self.T_cw[:3, 3]
        return -R.T @ t


@dataclass
class Landmark:
    id: int
    p_w: np.ndarray                                   # (3,)
    desc: np.ndarray                                  # (32,) descriptor of first observation
    obs: dict[int, int] = field(default_factory=dict)  # kf_id -> keypoint index


@dataclass(frozen=True)
class MapArrays:
    """Flat view of the map for the back-end. obs_* index into the kf_* / lm_* arrays."""

    kf_ids: np.ndarray        # (K,)
    kf_frame_idx: np.ndarray  # (K,)
    kf_t_ns: np.ndarray       # (K,) int64
    T_cw: np.ndarray          # (K, 4, 4)
    lm_ids: np.ndarray        # (L,)
    p_w: np.ndarray           # (L, 3)
    obs_kf: np.ndarray        # (M,) index into kf arrays
    obs_lm: np.ndarray        # (M,) index into lm arrays
    obs_uv: np.ndarray        # (M, 2) undistorted pixel coords


class Map:
    def __init__(self) -> None:
        self.keyframes: dict[int, Keyframe] = {}   # insertion order == time order
        self.landmarks: dict[int, Landmark] = {}
        self._next_kf = 0
        self._next_lm = 0

    def add_keyframe(self, frame_idx: int, t_ns: int, T_cw: np.ndarray, feats: Features) -> Keyframe:
        if self.keyframes:
            last = next(reversed(self.keyframes.values()))
            assert t_ns > last.t_ns, "keyframes must be added in time order"
        kf = Keyframe(self._next_kf, int(frame_idx), int(t_ns), np.array(T_cw, dtype=np.float64),
                      feats, np.full(len(feats), -1, dtype=np.int64))
        self.keyframes[kf.id] = kf
        self._next_kf += 1
        return kf

    def add_landmark(self, p_w: np.ndarray, desc: np.ndarray) -> Landmark:
        lm = Landmark(self._next_lm, np.asarray(p_w, dtype=np.float64).reshape(3).copy(), desc.copy())
        self.landmarks[lm.id] = lm
        self._next_lm += 1
        return lm

    def add_observation(self, lm_id: int, kf_id: int, kp_idx: int) -> None:
        kf, lm = self.keyframes[kf_id], self.landmarks[lm_id]
        if kf.lm_ids[kp_idx] >= 0:
            raise ValueError(f"keypoint {kp_idx} of kf {kf_id} already observes landmark {kf.lm_ids[kp_idx]}")
        if kf_id in lm.obs:
            raise ValueError(f"landmark {lm_id} already observed in kf {kf_id}")
        kf.lm_ids[kp_idx] = lm_id
        lm.obs[kf_id] = int(kp_idx)

    def last_keyframes(self, n: int) -> list[Keyframe]:
        """Up to n most recent keyframes, oldest first."""
        return list(self.keyframes.values())[-n:]

    def observations(self, lm_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(T_cw (V, 4, 4), uv (V, 2)) of every view of a landmark."""
        lm = self.landmarks[lm_id]
        kfs = [self.keyframes[k] for k in lm.obs]
        T = np.stack([kf.T_cw for kf in kfs])
        uv = np.stack([kf.feats.kps[i] for kf, i in zip(kfs, lm.obs.values())])
        return T, uv

    def track_lengths(self) -> np.ndarray:
        return np.array([len(lm.obs) for lm in self.landmarks.values()], dtype=np.int64)

    def to_arrays(self) -> MapArrays:
        kfs = list(self.keyframes.values())
        lms = list(self.landmarks.values())
        kf_index = {kf.id: i for i, kf in enumerate(kfs)}
        obs_kf, obs_lm, obs_uv = [], [], []
        for j, lm in enumerate(lms):
            for kf_id, kp in lm.obs.items():
                obs_kf.append(kf_index[kf_id])
                obs_lm.append(j)
                obs_uv.append(self.keyframes[kf_id].feats.kps[kp])
        return MapArrays(
            kf_ids=np.array([kf.id for kf in kfs], dtype=np.int64),
            kf_frame_idx=np.array([kf.frame_idx for kf in kfs], dtype=np.int64),
            kf_t_ns=np.array([kf.t_ns for kf in kfs], dtype=np.int64),
            T_cw=np.stack([kf.T_cw for kf in kfs]) if kfs else np.zeros((0, 4, 4)),
            lm_ids=np.array([lm.id for lm in lms], dtype=np.int64),
            p_w=np.stack([lm.p_w for lm in lms]) if lms else np.zeros((0, 3)),
            obs_kf=np.array(obs_kf, dtype=np.int64),
            obs_lm=np.array(obs_lm, dtype=np.int64),
            obs_uv=np.array(obs_uv, dtype=np.float64).reshape(-1, 2),
        )
