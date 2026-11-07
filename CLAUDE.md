# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ORB front-end + STEAP-style continuous-time trajectory estimation on pysteam, evaluated on
EuRoC MAV and compared against VGGT-Omega on the same images.

Two pipelines:
- **Monocular (cam0), M1-M9: done.** Kept as a baseline. Its main limits (M8): scale drift and
  tracking losses (no relocalisation), so ~60-90% coverage and ~0.6-0.9% of path ATE.
- **Stereo (cam0 + cam1, rectified), S1-S7: current work.** Metric scale removes scale drift
  and initialisation problems. The stereo VO in `../../Orb-feat-python/vo_euroc.py` is the
  reference front-end and the initialisation for STEAP.

Full design: the "STEAP Trajectory Estimation with an ORB Front-End" plan doc. This file
holds the rules and facts needed while coding; the plan doc holds the reasoning.

## Scope

- Estimation only. No STEAP planning factors (obstacle, goal costs).
- Cameras: stereo (`cam0` + `cam1`) for the stereo pipeline; `cam0` only for the monocular
  baseline. Do NOT use the IMU unless explicitly asked: everything stays image-only.
- VGGT-Omega comparison: VGGT-Omega was run monocular (cam0, stride 5). Comparing stereo
  STEAP against it is uneven; report it as such, or run VGGT-Omega with stereo images
  too (decision pending).
- Back-end is pysteam (`pip install asrl-pysteam`). Do not modify pysteam's source;
  new factors live in `pysteam_vo/backend/`.

## Layout

This repo's root *is* the `pysteam_vo` package (imports are `from pysteam_vo.data...`).
Run `pytest` and scripts from the parent directory (`ORB_based_pysteam/`) so
`import pysteam_vo` resolves. Paths below are relative to the repo root.

```
pysteam_vo/   (repo root)
  data/euroc.py              # EuRoC ASL loader -> images, timestamps (ns), K, dist, T_BC, GT
  frontend/orb.py            # ORB extractors (cv2 grid-bucketed | ORB-SLAM3 via python_orb_slam3), matching
  frontend/tracker.py        # mono: init, projection search + PnP, keyframes, triangulation, MonoVO
  frontend/map.py            # keyframes, landmarks, observation tracks -> MapArrays
  backend/pinhole_error.py   # PinholeErrorEvaluator (pysteam Evaluable)
  backend/problem.py         # BA (M5) and STEAP (M6) problems from MapArrays
  backend/cached_interp.py   # shared cached GP interpolation per frame
  backend/fast_problem.py    # vectorized Gauss-Newton assembly (same A, b as pysteam)
  backend/solver.py          # LM driver + Schur-complement LM solver
  eval/                      # align.py, export_tum.py, run_evo.sh, run_m*/plot_* drivers
  configs/default.yaml       # every threshold, noise value and qcd lives here
  results/                   # plots, TUM files (gitignored)
planned for stereo (S1-S3):
  data/stereo.py             # cam1 calib, rectification maps, rectified K / baseline, T_B_rect
  frontend/stereo_vo.py      # port of Orb-feat-python/vo_euroc.py + persistent landmark map
  backend/stereo_problem.py  # stereo factors (pysteam StereoErrorEvaluator) for BA / STEAP
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
- **Stereo camera frame = rectified left camera** (`rect`), not raw cam0. `cv2.stereoRectify`
  rotates cam0 by `R0`: `T_B_rect = T_B_C0 @ T_C0_rect` with `T_C0_rect[:3,:3] = R0.T`.
  Stereo knots are `T_rect,w`; stereo features live in rectified images with `K = P0[:3,:3]`
  and baseline `b = -P1[0,3] / P1[0,0]`. Never mix rectified pixels with raw-cam0 poses or K.
- **Stereo observation** = `(u_L, v_L, u_R)` from the rectified pair (`v_R = v_L`), which is
  what pysteam's `StereoErrorEvaluator` (`CameraIntrinsics(fu, fv, cu, cv, b)`) models.
- **Reference stereo VO output** (`Orb-feat-python`) is the *body* frame with world = first
  body pose. Convert to knots with `T_rect,w = inv(T_wb @ T_B_rect)`.

## pysteam gotchas

- `OptimizationProblem(use_sparse_matrix=True)` - the default is dense and will not
  scale past a few hundred landmarks.
- Robust loss: `CauchyLossFunc(1.0)` on reprojection terms. pysteam has no Huber.
- Gauge, monocular: lock the first knot pose (`SE3StateVar(T, locked=True)`), add a weak
  velocity prior on the first knot, and fix scale by locking one well-triangulated landmark
  from the initial pair.
- Gauge, stereo: lock the first knot pose only. Scale is observed through the baseline;
  do not lock a landmark.
- Large problems: use `SchurLevMarqSolver` (state order: all knot states, then 3-dim
  landmarks), `FastOptimizationProblem` and `CachedPoseInterpolator`. All three were checked
  to give pysteam's exact numbers; plain pysteam assembly/solve is 5-30x slower.
- `trajectory/const_vel` in the installed fork: the `jinv_velocity` Jacobian is a
  first-order approximation (0.3-2% off at our knot spacing). Do not modify pysteam.
- Observations from non-keyframe images attach via
  `traj.get_pose_interpolator(Time(nsecs=t))`; do not add knots for them.
- Model new evaluators on `pysteam/evaluable/stereo/stereo_error_evaluator.py`
  (`forward` returns the error; `backward` left-multiplies `lhs` by the Jacobian and
  passes it to the child).
- Reference usage: `tutorials/LandmarkSTEAMExample.ipynb` in the pysteam repo.

## EuRoC specifics

- Format: ASL folders. Images in `mav0/cam0/data/<timestamp_ns>.png` (752x480,
  grayscale, 20 Hz); list in `mav0/cam0/data.csv`.
- Read intrinsics, distortion and `T_BS` from `mav0/camX/sensor.yaml`; never hardcode
  them. Distortion is radial-tangential.
  - Monocular: undistort keypoint coordinates with `cv2.undistortPoints(..., P=K)` rather
    than undistorting whole images.
  - Stereo: rectify whole images (`cv2.stereoRectify` with `T_C1C0 = inv(T_B_C1) @ T_B_C0`,
    `CALIB_ZERO_DISPARITY`, `alpha=0`, then `initUndistortRectifyMap`). cam0/cam1 frames share
    timestamps; pair them by exact timestamp. Baseline is ~0.11 m.
- Ground truth: `mav0/state_groundtruth_estimate0/data.csv`. It is the IMU/body pose
  `T_wb` with the quaternion stored **w first** (qw, qx, qy, qz). Camera ground truth is
  `T_wc = T_wb @ T_bc`, where `T_bc` is cam0's `T_BS`.
- Ground truth starts and ends at slightly different times from the images: evaluate
  only on overlapping timestamps.
- MH_01 has a handheld phase (frames ~0-350) and a static period on the ground
  (~400-850) before take-off (~900). Monocular VO cannot initialise while static.
- Sequence difficulty for ORB: start with MH_01_easy and V1_01_easy.
  MH_04/05 (dark) and V1_03/V2_03 (fast, blurry) are expected to lose tracking;
  report that as a result, do not tune it away per sequence.

## Build order (do not skip ahead)

Each milestone must pass its check before starting the next. Fix pass criteria before
running; if one turns out to be wrong, say so and explain the change.

Monocular baseline (done):
1. M1 Loader: GT camera poses + triangulated points reproject under 1 px.
2. M2 Front-end on GT poses: triangulated landmarks match across views.
3. M3 VO: init + PnP only; follows GT after Sim(3) alignment.
4. M4 `PinholeErrorEvaluator`: Jacobian matches finite differences (rtol 1e-5);
   single-pose problem with GT landmarks fits the data at least as well as GT (Cauchy cost).
5. M5 Bundle adjustment in pysteam (no GP prior): ATE below M3.
6. M6 Add GP prior + non-keyframe observations; tune `qcd`.
7. M7 Interpolated poses match solved poses on held-out keyframes.
8. M8 Export + evo vs VGGT-Omega.
9. M9 ORB-SLAM3 extractor backend (`frontend.orb.backend`): mono coverage 72% +- 11% over
   5 runs vs 61% for cv2. Vocabulary/relocalisation/loop-closure milestones (M10-M14) are
   deferred; they may return later for stereo loop closure.

Stereo (current):
1. S1 Stereo rig + loader (done): rectification, rectified K / baseline, `T_B_rect`, GT
   rectified-left poses. Checks: raw OpenCV cam0/cam1 projections rectified (precise inverse,
   `cv2.undistortPointsIter`) equal our `(u_L, v, u_R)` model to 1e-3 px; unconstrained L/R
   matches |v_L - v_R| median < 1 px; pooled median stereo depth / GT two-view depth within
   1 +- 0.02 (got 0.993). Reprojecting stereo points into a later GT-posed frame is only a
   gross-error check (< 2 px): GT noise gives ~1.1 px even when correct, and wrong R0 or a 3%
   scale error score the same.
2. S2 Stereo VO port of `Orb-feat-python/vo_euroc.py`, outputting `T_rect,w` and flagging
   frames where tracking failed (no silent motion-model poses). Check: over 5 runs, ATE
   (SE(3)) and RPE match the reference script's within run-to-run spread (reference:
   0.643 m SE(3) ATE, 1.5 cm / 1 s RPE); every lost frame is flagged.
3. S3 Persistent stereo map (done; `StereoMapVO`): landmarks tracked across keyframes, tracking
   against the last 5 keyframes' landmarks by projection, S2-style fallback. Checks: GT-pose
   tracks (>= 3 KFs, multi-view re-triangulated) reproject median 0.49 / p90 0.95 px; 5-run
   VO: ATE SE(3) 0.588 +- 0.117 m vs S2 0.626 +- 0.166 (within spread), 10 s RPE 0.072 vs
   0.080 m. Drift direction is the same in every run (systematic component).
4. S4 Stereo factor: pysteam `StereoErrorEvaluator` composed with `ComposeLandmarkEvaluator`.
   Check: Jacobians match finite differences; single-pose solve against GT-posed landmarks
   fits at least as well as GT (Cauchy cost).
5. S5 Stereo BA (no GP prior), initialised from S3 VO. Check: keyframe ATE (SE(3)) below S3.
6. S6 Stereo STEAP: GP prior + non-keyframe stereo observations, initialised from VO (never
   from BA); re-tune `qcd` in metric units on one segment only. Check: all-frame ATE below
   VO; M7-style hold-out interpolation check.
7. S7 Export + evo: SE(3) and Sim(3) ATE/RPE vs GT, the reference stereo VO, monocular
   STEAP and VGGT-Omega, on identical timestamps.

## Testing

- From `ORB_based_pysteam/`: `pytest pysteam_vo/backend/tests` (and `pytest pysteam_vo/data/tests`) must pass before any change to the back-end is done.
- Every new evaluator gets a finite-difference Jacobian test.
- The GP setup gets a synthetic constant-velocity trajectory test (error near zero).
- Never report a result without running it; quote ATE/RPE numbers from evo output.
- The ORB-SLAM3 extractor is **not deterministic** (~1% of keypoints differ between calls on
  the same image: ORB-SLAM breaks ties by pointer address), and VO amplifies this. Report
  VO-dependent numbers as mean +- std over >= 5 runs, never from a single run. The cv2
  extractor is deterministic; single runs are reproducible but still only one sample.

## Evaluation

```
evo_ape tum gt_cam.tum steap.tum -as --plot --save_results results/steap_ape.zip
evo_rpe tum gt_cam.tum steap.tum -as --delta 1 --delta_unit m
evo_res results/*.zip --use_filenames
```

- Monocular: `-as` (Sim(3) alignment) is mandatory, since output is up to scale.
- Stereo: report `-a` (SE(3), no scale freedom) as the primary number, plus `-as` for
  comparison with monocular methods. Check that the Sim(3) scale is close to 1.
- Export a pose for every cam0 frame (interpolated), not just keyframes, so STEAP and
  VGGT-Omega are scored on identical timestamps.
- Log per run: ATE RMSE, RPE, recovered scale, fraction of frames tracked, runtime.

## Style

- Python 3.10+, NumPy, type hints on public functions.
- No magic numbers in code: thresholds and noise values go in `configs/default.yaml`.
- Keep the front-end free of pysteam imports; the back-end consumes plain arrays from
  `frontend/map.py`.
