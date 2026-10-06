# Calibration preparation with the short 40-degree ZED mount

Status: 30 poses and their approach/return paths have passed the **offline model
checks below**. Physical camera-to-robot calibration has not been performed.
No robot motion commands were sent. The live preview is separate from simulation.

## Current physical observations

The ZED 2i opened at HD1080/30 on the desktop. An eight-second stationary capture
saw no ChArUco board in the camera view. The live RGB-left/RGB-right/depth preview
was then restarted at http://127.0.0.1:8767, with stabilization off.

A read-only attempt to obtain current joints failed because the configured Ethernet
interface had no carrier. Simulation does not require that connection: it uses the
saved snapshot in `runs/camera-calibration/robot-snapshot.json`. That snapshot is
not a claim about the current robot position. Real calibration will need measured
joints synchronized with each image; commanding the robot is a separate action.

## Model and attachment assumptions

- Camera: `urdf/maver_x4_gp80_zed_40deg.urdf`, short 65 mm shelf datum, 40 degrees
  downward. Measured rectified-left camera intrinsics at 1920 x 1080.
- Pattern: 87 x 87 mm, 5 x 5 squares, 17.4 mm squares, 11.6 mm markers,
  DICT_4X4_50, IDs 0–11. Backing: 100 x 100 x 4 mm in the simulation; physical
  thickness/mass have not been measured.
- Board-to-right-flange transform is **estimated from the earlier photo**. At the
  saved hanging-hand pose, its printed face is assumed outward (+base X), with
  its top upward. Its centre is approximated as flange-local [0.02348191, 0, 0.16]
  metres plus 40 mm outward. The exact assumed matrix is saved in `route.json`.
  This estimate enables planning; it is not the hand-eye calibration result.
- Chassis, lift and left-arm references stay at the saved pose. Only right-arm
  references move. The physics servos show small residual deviations.
- Floor is modeled. The actual desks, cable, people and room are **not** modeled.

## Planning and verification

The initial regular Cartesian grid was rejected for reachability under the assumed
attachment. Instead, deterministic joint-space sampling generated 200 candidates
inside joint limits, with board projection/facing and geometric-clearance checks.
Of these, 168 passed actual ChArUco detection on rendered images with at least
12 corners. Thirty distinct observations were selected, with checked joint-space
connections. One intermediate approach waypoint was included. Return follows the
validated route in reverse.

Paths use quintic interpolation with zero endpoint velocity/acceleration, nominal
maximum joint speed 0.20 rad/s and acceleration 0.30 rad/s². Static path collision
checks sample at maximum L1 joint increment 0.005 rad. This is sampled geometric
validation, not a formal continuous-time collision certificate.

Collision checks use convex visual geometry, with same-weld and two-joint
mechanical housing-neighbor exclusions. The mounted board is checked against other
weld groups. Existing documented arm-seam exclusions are retained. Independent
AABB separation bounds avoid false zero distances from narrow-phase degeneracy.
These exclusions and conservative geometry do not establish hardware qualification.

The physics trial uses MuJoCo position servos and model gravity compensation,
**not the deployed MIT controller**. It integrates physics at 2 ms, logs measured
simulated joints at approximately 50 Hz, pauses two seconds at each observation,
and returns to the saved start. Rendering uses this measured simulation trace.

Results:

- 30/30 dwell images accepted, with 12–16 ChArUco corners.
- Maximum image reprojection RMS: 0.122 px.
- Full simulated approach/observations/return: 354.404 seconds.
- 17,831 physics clearance samples; minimum reported modeled clearance: 15.01 mm.
- Minimum right-joint limit margin: 0.04956 rad (2.84 degrees).
- Maximum dynamic right-joint tracking error: 0.005505 rad (0.315 degrees).
- All dwell tracking errors below 0.02 rad.
- No modeled collision detected; room/cable/attachment uncertainty excluded.
- Seventeen calibration regression tests passed.

The hand-eye solver was also exercised on these **synthetic rendered observations**:
24 fit poses and six preassigned held-out poses. Joint robust gradient descent
converged in 236 iterations. Held-out closure RMS was 0.366 mm / 0.214 degrees;
recovered camera transform differed from known simulation truth by 0.422 mm /
0.049 degrees. These numbers validate the synthetic pipeline, not the physical
camera's calibration accuracy. Do not install the synthetic transforms on hardware.

### Independent ground-truth recovery check

`tools/camera/verify_handeye_ground_truth.py` independently rerenders the 30
stationary observations and reruns detection, PnP and the existing hand-eye solver.
It reads reference transforms directly from MuJoCo body world poses and the
renderer camera pose (with explicit OpenGL-to-OpenCV axis conversion), instead
of using the planner's nominal camera transform to score the result. Neither
ground-truth unknown transform is passed to the solver or its initializer.

Two datasets use the same 24-fit/6-held-out split:

| Input | Camera-to-lift translation / rotation error | Board-to-hand translation / rotation error |
| --- | --- | --- |
| Exact MuJoCo poses | 2.90e-13 mm / 6.67e-15 degrees | 1.65e-13 mm / 2.35e-14 degrees |
| Rendered images → ChArUco → PnP | 0.4222 mm / 0.04892 degrees | 0.2415 mm / 0.04018 degrees |

Exact inputs are already solved by the Park initializer to floating-point
precision. The rendered case takes 236 refinement iterations; robust loss drops
from 1.29911 to 1.02136. Held-out closure RMS is 0.3661 mm / 0.2141 degrees.
Closure error and transform recovery error measure different quantities: the
former also includes each held-out PnP observation's error.

The verifier requires both exact transform errors below 1e-5 mm / 1e-5 degrees,
and both rendered transform errors below 1 mm / 0.2 degrees. Both cases passed.
These are simulation regression thresholds, not physical accuracy guarantees.
The four solver regression tests also passed, including noisy recovery,
Jacobian consistency and rejection of insufficient orientation diversity.

Reproduce with the simulation environment:

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/camera/verify_handeye_ground_truth.py
```

Outputs: `runs/camera-calibration-40deg/ground-truth-verification/`, including
`report.json` (full estimated and ground-truth matrices, errors and source hashes),
`observations.npz`, `detections.json` and all 30 raw rendered images.

## Artifacts and reproduction

All outputs are in `runs/camera-calibration-40deg/`:

- `initial-camera/`: actual camera connectivity/detection check.
- `sample/vision_candidates.json`: candidates filtered by rendered detection.
- `route/route.json`: 30 poses, approach/inter-pose/return segments and assumptions.
- `route/physics_trace.npz`, `physics_report.json`: measured simulation and checks.
- `route/calibration_simulation_6x.mp4`: 59.1-second H.264 review video, 6x playback.
- `route/calibration_30_pose_views.jpg`: board-view contact sheet.
- `route/pose_XX_camera.png`: full-resolution detected simulation images.
- `route/synthetic_handeye_check.json`: simulation-only solver results.
- `route/artifact-validation.json`: video decode check and source hashes.

From this checkout, using the existing simulation Python and `MUJOCO_GL=egl`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`:

```bash
python tools/camera/validate_calibration_poses.py \
  --snapshot runs/camera-calibration/robot-snapshot.json \
  --output runs/camera-calibration-40deg/sample
python tools/camera/filter_calibration_views.py
python tools/camera/plan_calibration_route.py
python tools/camera/simulate_calibration_route.py
python tools/camera/render_calibration_route.py
```

These scripts are offline-only. They do not import hardware endpoints. Do not
pass the pose file to a live controller as an authorization artifact.

## Before physical capture

Review the video against the actual board attachment and workspace. Obtain a
stationary bootstrap board observation paired with measured joints, verify the
coarse board placement, and update the modeled starting state and obstacles.
Revalidate the resulting approach and the actual deployed controller/payload path.
Then collect synchronized stationary board detections and measured FK for all
poses, fit on the 24 assigned samples and evaluate the six held-out samples.
Solve camera-to-lift and board-to-right-flange; compose lift FK for camera-to-base.
The camera remains on the moving lift, so camera-to-base changes with lift height.
