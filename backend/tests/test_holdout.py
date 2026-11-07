"""M7 checks: GP-interpolated poses vs solved poses on held-out keyframes.

Synthetic tests always run. The EuRoC test is marked `slow` (two STEAP solves).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysteam_vo.backend import problem as P
from pysteam_vo.backend.solver import solve_lm
from pysteam_vo.backend.tests.test_steap import K, _accel_scene, _cv_scene, _perturb
from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import EurocSequence, interpolate_poses, invert_se3
from pysteam_vo.eval.run_m7_holdout import holdout_split

CFG = load_config()
BE = CFG["backend"]
M7 = CFG["m7_check"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")


def test_holdout_split_moves_observations_not_drops_them():
    arr, nonkf, _ = _cv_scene(n_kf=8)
    arr_ho, nonkf_ho, held = holdout_split(arr, nonkf, every=3)
    assert held.tolist() == [3, 6]
    assert arr_ho.kf_t_ns[0] == arr.kf_t_ns[0] and arr_ho.kf_t_ns[-1] == arr.kf_t_ns[-1]
    assert len(arr_ho.obs_kf) + len(nonkf_ho[0]) == len(arr.obs_kf) + len(nonkf[0])
    moved_t = set(nonkf_ho[0].tolist()) - set(nonkf[0].tolist())
    assert moved_t == set(arr.kf_t_ns[held].tolist())
    np.testing.assert_array_equal(arr_ho.T_cw, arr.T_cw[np.setdiff1d(np.arange(8), held)])


def _solve_holdout(arr, nonkf, held_every, cfg):
    full = P.build_steap_problem(arr, nonkf, K, cfg)
    arr_ho, nonkf_ho, held = holdout_split(_perturb(arr, full.anchor_row), nonkf, held_every)
    ho = P.build_steap_problem(arr_ho, nonkf_ho, K, cfg, landmark_rows=full.lm_rows, anchor_row=full.anchor_row)
    solve_lm(ho.problem, cfg, n_pose=ho.n_pose_dims)
    return ho, arr_ho, held


def test_holdout_interpolation_exact_on_constant_velocity():
    arr, nonkf, _ = _cv_scene(n_kf=8)
    cfg = dict(BE, relative_cost_change_threshold=1e-12, absolute_cost_change_threshold=1e-16,
               gp=dict(BE["gp"], first_velocity_prior_sigma=1e6))
    ho, _, held = _solve_holdout(arr, nonkf, 3, cfg)
    np.testing.assert_allclose(ho.interpolate(arr.kf_t_ns[held]), arr.T_cw[held], atol=1e-6)


def test_gp_interpolation_beats_linear_on_accelerating_motion():
    arr, nonkf, _ = _accel_scene(n_kf=10, noise_px=0.5)
    ho, arr_ho, held = _solve_holdout(arr, nonkf, 3, BE)
    t = arr.kf_t_ns[held]
    T_lin = invert_se3(interpolate_poses(arr_ho.kf_t_ns, invert_se3(ho.poses()), t, 2**62)[0])
    c = lambda T: invert_se3(T)[:, :3, 3]
    e_gp = np.linalg.norm(c(ho.interpolate(t)) - c(arr.T_cw[held]), axis=1)
    e_lin = np.linalg.norm(c(T_lin) - c(arr.T_cw[held]), axis=1)
    print(f"\nheld-out pose error: GP {1e3 * np.median(e_gp):.2f} mm, linear {1e3 * np.median(e_lin):.2f} mm")
    assert np.median(e_gp) < np.median(e_lin)


@needs_data
@pytest.mark.slow
def test_m7_interpolation_matches_solved_poses():
    from pysteam_vo.eval.run_m7_holdout import report, run_m7

    seq = EurocSequence.from_config(CFG)
    r = run_m7(seq, CFG, M7["start_frame"], M7["end_frame"], M7["holdout_every"])
    print()
    report(r, seq.name, M7["start_frame"], M7["end_frame"])
    dt, dr = r["gp_vs_solved_local"]
    lt, lr = r["lin_vs_solved_local"]
    assert np.median(dt) < M7["max_median_trans_m"]
    assert np.median(dr) < M7["max_median_rot_deg"]
    assert np.median(dt) < np.median(lt) and np.median(dr) <= np.median(lr)
