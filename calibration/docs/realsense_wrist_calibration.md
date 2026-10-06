# D435i right-wrist calibration — fixed 33° mount

This workflow estimates the right wrist's **RGB optical extrinsic** using the
stationary 87 × 87 mm ChArUco print on the table. It does not reuse or overwrite
the head camera calibration. The target camera is D435i serial `243222074562`.

## Setup and operation

Keep the board flat and firmly attached to the table, with the complete print
visible. Leave the board and mobile base stationary for the entire run. The
pattern is five by five squares: square length `0.0174 m`, marker length
`0.0116 m`, dictionary `DICT_4X4_50`. The 87 mm dimension refers to the patterned
area, excluding its white border. Keep the white border intact.

The current route assumes the fixed 33° one-piece mount and the board location
observed by the head camera on October 6, 2026. That head observation is a rough
planning seed, not calibration evidence. Its estimated patterned-area center in
the URDF base frame is approximately `(0.722, 0.033, 0.725) m`. The table footprint
and height in the simulator are approximate. Inspect the rendered route against
the actual table, mount and cables before starting physical movement. If the
board or table has moved, regenerate/review the route first.

**Mount-side correction, October 6:** The camera is on the **outside of the
right arm at default**, as shown in the supplied mounting photos. The complete
CAD assembly is installed with a 180° rotation about the flange Z axis; this
rotation applies to the supporter, housing and RGB optical frame together.
The fixed 33° tilt and original STL remain unchanged. Installation is declared
in `config/camera/right_fixed33_installation.json`. The camera pupil is 57.47 mm
outward from the right wrist at the 300-mm default lift in the nominal model.
The original inward-side route and its videos are superseded for this physical
mount. The launcher now selects the regenerated outside-side plan and rejects
the old route.

Run from the consolidated calibration directory:

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
./tools/camera/run_wrist_calibration.sh --mode live --confirm-onsite-ready
```

Open **http://127.0.0.1:8772** and click **Start calibration** once. Startup first
performs the existing read-only robot audit. The button then performs the
controlled upper-lift retreat if required, restores both arms and the lift to
the default pose (`300 mm` lift), and attempts all 30 right-arm poses. The base,
lift and left-arm references remain fixed during collection. The page shows the
pose ID, attempted count, eligible count and reasons for rejected images.

**Pause** (Space) requests a guarded stop and pauses automatic advance. Wait for
stationary HOLD, then click **Start/Resume calibration** to continue. The
**Move to default pose** button includes the lift; **Return arms to default**
preserves lift height. The route uses checked intermediate waypoints on return,
because a direct joint interpolation can dip toward the tabletop. Browser loss
or a hardware feedback/control fault requests a stop; restarting motion requires
an explicit operator action. Use the physical E-stop for an emergency.

After all attempts, fitting saves a candidate result **before** the guarded arm
return. A failed return cannot erase the captured dataset or completed fit. No
camera transform is automatically applied to deployment.

The calibration process owns the camera exclusively; stop a separate RealSense
viewer before launching. Native RGB is `1280 × 720 @ 30 Hz`. The saved filename
`left.png` is retained for compatibility with the old IPC; it contains the D435i
**color sensor** image, not a stereo-left image.

## What is fitted

Transforms named `T_A_from_B` convert coordinates in frame B into frame A.
For observation i:

```text
G_i = T_base_from_wrist(q_measured_i)       known from robot FK
C_i = T_color_optical_from_board_i         known from ChArUco PnP
X   = T_wrist_from_color_optical           unknown, constant
Y   = T_base_from_board                    unknown, constant

G_i · X · C_i = Y
```

The robot base is `maver_x4_asm_maver_x4_base_link`, the same base frame used by
the URDF/IK. The right-wrist parent is `maver_x4_asm_mount2`. The optical frame is
`realsense_243222074562_color_optical`, with x right, y down, z forward.

The solver initializes with relative-motion hand-eye calibration, then jointly
refines both rigid transforms on SE(3) using a robust gradient optimization.
The wrist mount matrix stays fixed. The camera pose in the base frame changes
with robot joints, including lift height:

```text
T_base_from_camera(q) = FK_base_from_mount2(q) · T_parent_from_optical
```

RGB intrinsics are read from the SDK for the actual stream profile. This camera's
color distortion coefficients are zero. Factory depth-to-color extrinsics are
not fitted by this workflow. Depth is not required for the hand-eye fit.

## Capture and validation

Capture requires fresh, stationary measured feedback and up to 5° target offset
per right-arm joint. Target angles are not substituted for encoders in FK. The
worker maps the SDK GLOBAL_TIME exposure timestamp to host monotonic time,
records its uncertainty, and rejects unreliable timing. Stationary pose pairing
is used; this is not hardware-trigger synchronization.

Images require at least 12 detected ChArUco corners and the existing detection
quality checks. Poor images are saved as **not eligible**, and unreachable poses
as **invalid**; after confirmed HOLD, collection continues. Hardware faults and
failed guarded preparation/return remain stops. The 5° capture allowance does
not loosen robot limits or default/preparation arrival checks.

At least 12 eligible observations are required. Every fifth eligible observation
is held out from fitting. Reports include translation/rotation closure errors
and raw-corner reprojection error on held-out poses. A converged fit is a
candidate requiring independent physical validation, such as measured-joint
real/simulation arm overlays. A simulation result is not a physical calibration.

## Files and reproduction

Each launch creates `runs/wrist-calibration/<UTC timestamp>/` containing
`dataset.json`, camera metadata, per-attempt RGB images, pairing/eligibility
records, motion-path checks, the deployment binding and versioned
`extrinsics-*.json`. The result includes `T_parent_from_optical`,
`T_wrist_from_camera` and `T_base_from_board`, with explicit frame names.

Saved data can be fitted again without a robot connection, including after a
later physical/control interruption:

```bash
../../../.venv/bin/python tools/camera/solve_wrist_calibration.py \
  runs/wrist-calibration/<session>/dataset.json \
  --output runs/wrist-calibration/<session>/extrinsics-offline.json
```

Rehearse the entire automatic controller/camera workflow without hardware:

```bash
./tools/camera/run_wrist_calibration.sh --mode sim --auto
```

Score a new end-to-end simulated session against its independent MuJoCo camera
and board transforms:

```bash
../../../.venv/bin/python tools/camera/verify_wrist_ground_truth.py \
  runs/wrist-calibration/<simulated-session> \
  --output runs/wrist-calibration/<simulated-session>/ground-truth-recheck.json
```

The corrected checked plan is
`config/camera/realsense_d435i_right_fixed33_outside_plan.json`. Planning render
sheets and reports are in `runs/right-wrist-outside/planning/`. All 30 nominal
views detected 14–16 ChArUco corners with the closed planning gripper. Default
preparation, connecting segments and
return were sampled at no more than a 0.005 sum-of-joint-coordinate increment,
over 13,530 samples, with minimum modeled clearance **15.001 mm** and 0.15 rad
target arm-joint reserve. This is geometric verification; it does not establish
motor dynamics or the true room geometry. Runtime paths are rechecked from both
measured joints and the retained command reference.

This mount correction has not yet had a new end-to-end controller rehearsal or
physical calibration. The previous migrated rehearsal attempted 30 poses with
27 eligible images, camera ground-truth error 0.209575 mm / 0.056891°, and a
held-out RMS of 0.124810 px. Those figures belong to the old inward-side geometry
and remain historical solver evidence only. They do not validate the installed
outside mount. Raw historical runs are preserved.

The original route validation retained a rigid, closed gripper in renders.
At fixed 33°, its fingers obstruct the center ray, so the planned board views
lie near `(640, 170)` in the RGB image.
Only the camera's opaque housing proxy is hidden from its own render; it remains
in collision checks and third-person views.

To regenerate the plan from the local CAD, recorded camera intrinsics and
read-only robot snapshot:

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../../.venv/bin/python \
  tools/camera/wrist_calibration_geometry.py --output runs/right-wrist-outside/planning
```

The plan checks hashes of its source model, mapping, CAD and planning inputs.
Both arms retain the deployed J6 correction; changing the mapping or geometry
requires fresh plan verification.

## Open-gripper visibility review

The user requested opening the right GP80 for a clearer wrist-camera view.
`tools/camera/render_wrist_open_gripper.py` separates the source CAD's palm and
two jaws and uses the parent's approximate parallel-slide model: 40 mm per jaw
in opposite directions, nominal 80 mm opening. It keeps the fingers, palm,
camera mount and third-person housing visible. Source/derived mesh hashes are
saved in the report; the original scenes and source meshes are preserved.

Run in the shared MuJoCo/OpenCV/trimesh environment with PyAV installed
(`hexfellow-deploy[calibration-video]` adds PyAV):

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../../.venv/bin/python tools/camera/render_wrist_open_gripper.py \
  --source-scene runs/right-wrist-outside/planning/geometry/scene.xml \
  --poses-only --output runs/right-wrist-outside/open-gripper
```

The corrected outside-side review detects all 16 ChArUco corners in all 30
nominal target images. The center camera ray is clear in all 30; gripper image
coverage averages 27.94% closed and 15.93% open. Output includes a default-pose
mount close-up, `closed-vs-open.png`, a 60-second nominal-pose MP4, and
`visibility-report.json`. The 1280×720 H.264 video is fully decoded and its
sample frames visually inspected. These are static target renders with an
explicit jaw override, not measured feedback or a new controller rehearsal.
Open-jaw trajectory clearance and physical operation have not been qualified.

After recording a new outside-mount simulated session, omit `--poses-only` and
pass its directory as `--sequence-run` to render measured **simulation**
`state.q` for arm/lift with the jaw override. The renderer checks that its scene
matches the active installation and will not silently reuse an inward-side scene.
The earlier `runs/right-wrist-open-gripper/` movies are retained as superseded
inward-side evidence.
