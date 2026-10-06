# Independent left-hand camera validation

The [complete calibration guide](camera_calibration.md) records the latest robust
landmark PnP result, current matrices and versioned numerical inputs. The
historical overlay checks below used the earlier coarse board estimate; the live
launcher still references that estimate until its route is regenerated and reviewed.

The current launcher uses the `both_j6` mapping: left J6 and right J6 both have
sign -1. Its converted default retains the same SDK motor positions. The three
validation poses are rechecked in simulation for this mapping. Historical
`right_j6` captures must be remapped, not replayed as corrected model coordinates.
The offline tool `tools/camera/replay_joint_mapping.py` writes separate corrected
overlays with both mapping hashes and keeps the original captures unchanged.
Use `--whole-arm` to include every visible left-arm link and the gripper in the
offline overlay. The validation website now uses whole-arm overlays by default.

## Run on the robot

Close the previous calibration/robot-control process so it releases the existing
onsite lock. Keep the ZED connected to the desktop, the 40-degree mount fixed,
and the robot workspace and USB cable clear. Then run:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
./tools/camera/run_left_hand_validation.sh --mode live --confirm-onsite-ready
```

After the existing read-only startup audit passes, open
**http://127.0.0.1:8773** and click **Start validation** once. The sequence:

1. Checks the approach from the actual robot state and moves both arms to the
   existing default pose, with lift at 300 mm.
2. Moves the left hand through the three simulated poses. The right hand and
   lift retain their current positions during these three movements.
3. At each stationary HOLD, captures a full-resolution rectified LEFT image and
   pairs it with measured joint feedback. No board detection is required.
4. Returns the arms to default, then renders the robot at each captured measured
   configuration and displays **real / simulation / overlay** for all captures.

The page shows `Pose 1 / 3` through `Pose 3 / 3`. **Pause** or Space stops the
sequence and requests controlled stopping; **Start** resumes with a newly
checked path. **Return arms to default** interrupts the sequence and checks a
return path. Keep the page open during movement. **Stop & quit** releases the
session after reviewing results. Physical E-stop remains available.

The active left arm may have up to 5 degrees of target error per joint for a
capture, provided it is stationary and the measured configuration passes the
geometry check. Replay uses those measured angles. A rejected/unreachable target
is recorded as invalid and skipped after stationary HOLD. Feedback, power,
controller, or unsafe-current-state faults retain the deployment's stopping
behavior. Existing right-arm calibration thresholds remain unchanged.

Each run prints its output directory:

```text
runs/left-hand-validation/<timestamp>/
  dataset.json                 # Camera intrinsics, measured q and time pairing
  camera/<capture-id>/left.png  # Full-resolution real images
  comparisons/index.html       # Open after quitting to view all comparisons
  comparisons/pose-01/          # real.png, sim.png, overlay.png, comparison.png
  comparisons/pose-02/
  comparisons/pose-03/
  comparisons/summary.json
  result.json
```

The report is a visual validation, not an automatic declaration of extrinsic
accuracy. If a pose is invalid, its reason appears on the page and in the saved
dataset. A hand that is unexpectedly outside the actual view is still saved for
inspection instead of being filtered by ChArUco detection.

For the same workflow without any physical robot access:

```bash
./tools/camera/run_left_hand_validation.sh --mode sim
```

Render the robot at **measured model joint coordinates**, using the fitted
`T_mount_from_camera` and the ZED's rectified LEFT intrinsics. Compare the hand
silhouette with the same physical hand in the real image. No image translation,
homography, optical flow, or extrinsic refitting is allowed during validation.

The calibration used the right hand. New left-hand observations are therefore
an independent check of the camera/robot alignment, although a discrepancy can
also come from left-arm mapping, FK, flex, or gripper geometry. This does not
isolate camera error by itself.

## Prepared and tested

All paths below are relative to this checkout.

- Candidate fit:
  `runs/result-analysis/20260930T220448.229707Z/four-corner-handeye-diagnostic.json`.
  It remains a candidate, not an applied or hardware-validated transform.
- Existing real-image replay:
  `runs/camera-overlay-validation/right-held-out-06/comparison.png`.
  The right arm broadly agrees visually. This is not the new left-hand test.
- Three left-hand poses:
  `runs/camera-overlay-validation/left-runtime-plan/plan.json`.
- Deployment-controller simulation:
  `runs/camera-overlay-validation/left-dynamics/report.json` and
  `left-hand-validation-poses.png` in the same directory.

The simulation completed all three poses and returned to its configured default
at lift 0.300 m. The rendered gripper is near the image center, at roughly
0.50–0.60 m camera depth. Its measured simulated joint errors were at most
0.0054 degrees. All 2,812 sampled simulation states and intervening segments
passed the existing 15 mm modeled-clearance check. The actual room, desk, cables,
and people are not modeled. The start is a configured simulation pose, not a
measurement of the present robot. These files are not a live motion command.

The fitted optical center differs from nominal 40-degree CAD by approximately
[-55.0, -7.0, -16.7] mm in the lift frame, with 1.19 degrees rotation difference.
Consequently the nominal support mesh intrudes into the fitted simulated camera
view. Do not treat that patch as a predicted real occlusion or silently remove it
to claim agreement. The left gripper is outside it in the three planned views.
The camera-support CAD placement needs independent checking too.

## Physical comparison procedure

1. Keep the camera mount fixed and use the same mapped joint convention as the
   calibration: current `both_j6` deployment mapping, including joint signs and
   offsets. The saved `q` vector has base X/Y/yaw, lift, left J1–J7, right J1–J7;
   it is not raw SDK encoder angles.
2. Through the operator-controlled deployment stack, check the approach from the
   actual starting configuration before moving to each reviewed left-hand pose.
   Do not inject the plan directly into the SDK or reuse a default-origin path
   when the real arm is elsewhere.
3. In stationary HOLD, save a full-resolution rectified LEFT image, its current
   K, timestamps, and measured joints bracketing the exposure. Reuse
   `calibration_runtime.stationary_pair` for the same freshness, stationarity,
   and host-time pairing checks as calibration. This test does **not** require
   ChArUco detection or a minimum board-corner count. The launcher above performs
   these capture and pairing steps automatically.
4. The launcher records each sample in the existing dataset shape: `camera` metadata and an
   `observations` row with `pose_id`, `image`, `q`, and `pairing`. The image must
   match the resolution of its K. In the replay CLI, use `--side left`.
5. Inspect the overlaid housing/wrist edges across all three views. Keep the
   physical finger opening consistent with the fixed gripper mesh. Do not score
   background, orange cables, or the other hand's board as hand geometry.
6. For a number rather than visual judgment, manually annotate the **visible**
   real gripper mask at the original image size. Supply `--real-mask mask.png`:
   white=gripper, black=background. The tool reports intersection-over-union and
   symmetric boundary mean/P95 pixel error. Without that independent annotation
   it deliberately reports no numerical overlap score. Model/real occlusions
   and finger-opening differences need to be accounted for when interpreting it.

Because the camera is on the lift, the renderer attaches the fitted transform
to the lift body, not to the world or a constant base pose. Measured lift motion
therefore updates the camera automatically. Accurate arrival at the requested
joint angles is not needed for this comparison: use the stationary measured
angles, with the original motion/feedback guards still in force.

## Reproduce the saved-image comparison

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
../../../.venv/bin/python tools/camera/validate_camera_overlay.py \
  --dataset runs/live-calibration/20260930T220448.229707Z/dataset.json \
  --extrinsics runs/result-analysis/20260930T220448.229707Z/four-corner-handeye-diagnostic.json \
  --pose 6 --side right --whole-arm \
  --output runs/camera-overlay-validation/right-replay-new
```

The output directory must be new. Outputs include the real image, simulated
image, blended silhouette overlay, predicted mask, comparison figure, and JSON
with measured joints, intrinsics, fitted transform, pairing metadata and hashes.

To regenerate the offline left plan and nominal dynamics check:

```bash
../../../.venv/bin/python tools/camera/plan_left_hand_validation.py \
  --dataset runs/live-calibration/20260930T220448.229707Z/dataset.json \
  --extrinsics runs/result-analysis/20260930T220448.229707Z/four-corner-handeye-diagnostic.json \
  --output runs/camera-overlay-validation/left-plan-new
../../../.venv/bin/python tools/camera/verify_left_hand_validation.py \
  --plan runs/camera-overlay-validation/left-plan-new/plan.json \
  --output runs/camera-overlay-validation/left-dynamics-new
```

Both commands are simulation-only. The verifier explicitly rejects hardware
gateway construction and runs the existing deployment controller with nominal
MuJoCo MIT dynamics. It does not change any live safety or arrival thresholds.

## Workflow verification

The operator-started simulation passed Start, Pause, Resume, all three captures,
return to default, display of all three overlays, and Quit. No physical robot
commands were sent. Evidence is in
`runs/camera-overlay-validation/operator-workflow-final/verification-summary.json`
and `ui/ui-report.json` in that run. All 114 calibration tests passed.
