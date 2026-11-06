# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Monocular ORB front-end + STEAP-style continuous-time trajectory estimation on pysteam,
evaluated on EuRoC MAV. Results will be compared against VGGT-Omega on the same images.

Full design: the "STEAP Trajectory Estimation with an ORB Front-End" plan doc. This file
holds the rules and facts needed while coding; the plan doc holds the reasoning.

## Scope

- Estimation only. No STEAP planning factors (obstacle, goal costs).
- Monocular: EuRoC `cam0` only. Do NOT use the IMU or `cam1` unless explicitly asked;
  the VGGT-Omega comparison must be image-only on both sides.
- Back-end is pysteam (`pip install asrl-pysteam`). Do not modify pysteam's source;
  new factors live in `pysteam_vo/backend/`.

## Layout

This repo's root *is* the `pysteam_vo` package (imports are `from pysteam_vo.data...`).
Run `pytest` and scripts from the parent directory (`ORB_based_pysteam/`) so
`import pysteam_vo` resolves. Paths below are relative to the repo root.

```
pysteam_vo/   (repo root)
  data/euroc.py            # EuRoC ASL loader -> images, timestamps (ns), K, dist, T_BC, GT
  frontend/orb.py          # ORB extraction (grid-bucketed) + matching
  frontend/tracker.py      # init, PnP tracking, keyframes, triangulation
  frontend/map.py          # keyframes, landmarks, observation tracks
  backend/pinhole_error.py # PinholeErrorEvaluator (pysteam Evaluable)
  backend/problem.py       # knots, landmarks, cost terms from the map
  backend/solver.py        # batch LM driver
  backend/tests/
  eval/export_tum.py
  eval/run_evo.sh
  configs/default.yaml     # every threshold, noise value and qcd lives here
  run.py
```

## Conventions (most bugs come from here)

- **Poses.** pysteam knot pose is `T_k0` = camera-from-world (`T_cw`). This is what
  `cv2.solvePnP` returns. Ground truth and TUM exports are world-from-camera (`T_wc`).
  Convert only at the loader and the exporter; name every pose variable `T_<to><from>`
  (e.g. `T_cw`, `T_wc`, `T_wb`, `T_bc`).
- **Twists.** pylgmath / pysteam order se(3) vectors as (translation, rotation) - the
  reverse of GTSAM. `qcd`, velocities and noise covariances follow this order.
- **Points.** Landmarks are 4x1 homogeneous `HomoPointStateVar`, last entry 1.
- **Time.** Keep timestamps as int64 nanoseconds everywhere. Build pysteam times with
  `Time(nsecs=t_ns)`, never `Time(secs=t_ns * 1e-9)` (EuRoC stamps are ~1.4e18 ns and
  lose precision as floats).
- **Knots are camera poses.** Body/IMU poses appear only in the loader, where ground
  truth is converted to the camera frame.

## pysteam gotchas

- `OptimizationProblem(use_sparse_matrix=True)` - the default is dense and will not
  scale past a few hundred landmarks.
- Robust loss: `CauchyLossFunc(1.0)` on reprojection terms. pysteam has no Huber.
- Gauge: lock the first knot pose (`SE3StateVar(T, locked=True)`), add a weak velocity
  prior on the first knot, and fix monocular scale by locking one well-triangulated
  landmark from the initial pair.
- Observations from non-keyframe images attach via
  `traj.get_pose_interpolator(Time(nsecs=t))`; do not add knots for them.
- Model new evaluators on `pysteam/evaluable/stereo/stereo_error_evaluator.py`
  (`forward` returns the error; `backward` left-multiplies `lhs` by the Jacobian and
  passes it to the child).
- Reference usage: `tutorials/LandmarkSTEAMExample.ipynb` in the pysteam repo.

## EuRoC specifics

- Format: ASL folders. Images in `mav0/cam0/data/<timestamp_ns>.png` (752x480,
  grayscale, 20 Hz); list in `mav0/cam0/data.csv`.
- Read intrinsics, distortion and `T_BS` from `mav0/cam0/sensor.yaml`; never hardcode
  them. Distortion is radial-tangential: undistort keypoint coordinates with
  `cv2.undistortPoints(..., P=K)` rather than undistorting whole images.
- Ground truth: `mav0/state_groundtruth_estimate0/data.csv`. It is the IMU/body pose
  `T_wb` with the quaternion stored **w first** (qw, qx, qy, qz). Camera ground truth is
  `T_wc = T_wb @ T_bc`, where `T_bc` is cam0's `T_BS`.
- Ground truth starts and ends at slightly different times from the images: evaluate
  only on overlapping timestamps.
- Sequence difficulty for monocular ORB: start with MH_01_easy and V1_01_easy.
  MH_04/05 (dark) and V1_03/V2_03 (fast, blurry) are expected to lose tracking;
  report that as a result, do not tune it away per sequence.

## Build order (do not skip ahead)

Each milestone must pass its check before starting the next.

1. M1 Loader: GT camera poses + triangulated points reproject under 1 px.
2. M2 Front-end on GT poses: triangulated landmarks match across views.
3. M3 VO: init + PnP only; follows GT after Sim(3) alignment.
4. M4 `PinholeErrorEvaluator`: Jacobian matches finite differences (rtol 1e-5);
   single-pose problem with GT landmarks recovers the GT pose.
5. M5 Bundle adjustment in pysteam (no GP prior): ATE below M3.
6. M6 Add GP prior + non-keyframe observations; tune `qcd`.
7. M7 Interpolated poses match solved poses on held-out keyframes.
8. M8 Export + evo vs VGGT-Omega.

## Testing

- From `ORB_based_pysteam/`: `pytest pysteam_vo/backend/tests` (and `pytest pysteam_vo/data/tests`) must pass before any change to the back-end is done.
- Every new evaluator gets a finite-difference Jacobian test.
- The GP setup gets a synthetic constant-velocity trajectory test (error near zero).
- Never report a result without running it; quote ATE/RPE numbers from evo output.

## Evaluation

```
evo_ape tum gt_cam.tum steap.tum -as --plot --save_results results/steap_ape.zip
evo_rpe tum gt_cam.tum steap.tum -as --delta 1 --delta_unit m
evo_res results/*.zip --use_filenames
```

- `-as` (Sim(3) alignment) is mandatory: monocular output is up to scale.
- Export a pose for every cam0 frame (interpolated), not just keyframes, so STEAP and
  VGGT-Omega are scored on identical timestamps.
- Log per run: ATE RMSE, RPE, recovered scale, fraction of frames tracked, runtime.

## Style

- Python 3.10+, NumPy, type hints on public functions.
- No magic numbers in code: thresholds and noise values go in `configs/default.yaml`.
- Keep the front-end free of pysteam imports; the back-end consumes plain arrays from
  `frontend/map.py`.
