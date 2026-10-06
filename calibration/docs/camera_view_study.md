# ZED mounting visibility study — 2026-09-29

**Correction:** the initial study below used the older tall v3 bracket. It does not
represent the shortened supporter. The corrected integrated model uses short
triangular v5, a 65 mm shelf datum and 15-degree downward pitch. Corrected images
and metrics are in `runs/camera-view-study-short-v5/`; see the integrated-model
section below. Earlier 32/42/55-degree conclusions are historical comparisons.

The nominal 32-degree PoolBot camera view loses the near tabletop grasp area when
rendered with this physical ZED's rectified 1080p intrinsics. Tilting down improves
close manipulation, but a steep fixed angle loses the far part of a billiard table.
A 42-degree downward orientation is a useful overview candidate; 55 degrees is a
close-table candidate. These are scenario-dependent design candidates, not measured
extrinsics or an instruction to move the robot.

## Evidence and assumptions

- Branch `zi/dev_camera`, isolated checkout `UMI_Hardware_camera`; original `zi/dev`
  and the PoolBot assets were not modified.
- ZED 2i serial 32701991, measured SDK rectified LEFT camera, 1920 x 1080:
  fx = fy = 1055.564819 px, cx = 980.437378, cy = 559.544983.
  Actual image angular coverage: **84.559 x 54.174 degrees**.
- PoolBot `configs/default.yaml` uses **110 degrees horizontally** and 32 degrees
  down. Its field of view is wider than this measured rectified output. The legacy
  comparison is deliberately labeled; it must not be treated as the real camera.
- Mount offset/tilt come from PoolBot's `poolbot.urdf`, not physical calibration.
  The user photos show a camera above the moving shoulder/lift plate. The actual
  installed pitch is still unknown. A level-camera case brackets that uncertainty.
- Lift was measured at 0.421999 m; nominal left optical center height is 1.56 m.
  The lower variant lowers the **entire lift by 20 cm**, not only the bracket.
- The 7-foot table uses PoolBot's 1.9812 x 0.9906 m cloth, 0.79 m height,
  and 57.15 mm balls. Rails/pockets and workbench objects are simplified geometry.
- All frames are actual MuJoCo raster renders of static robot poses. GP80 meshes
  and measured starting joints are retained; bench poses use offline position IK.
  No hardware commands or dynamic trajectories were executed.
- Requested conversation `01a0e962-d54e-7cd1-93af-d1381ce4fc9b` could not be read:
  the provider rejected chat route access (403). Local PoolBot assets were used.

## Scenarios

Positions are relative to the robot base origin, with +X forward.
The bench is x=0.42..1.27 m, y=+-0.40 m, top z=0.76 m. Grasp and transfer
show distinct right-arm configurations; the arm can occlude part of the surface.
The end-on pool near cloth edge is x=0.60 m, approximately 0.44 m horizontally
from the nominal camera. The backed-up end case places it at x=1.00 m (~0.84 m
from the camera). The side-on cases put the long dimension across the image:
near cloth edge x=0.60 m or x=1.30 m (~1.14 m from the camera).

| Scene | Nominal 32° | 42° candidate | 55° candidate |
|---|---|---|---|
| Bench grasp | Near object cropped; 3/4 object targets visible | All 4 targets visible; near table edge cropped | All 4 targets and all surface corners in frame |
| Right-hand transfer | Near region cropped | All 4 object targets visible | Full surface rectangle in frame; arm still occludes some area |
| Pool, close end | Near cloth corners cropped | All 4 cloth corners and all 16 ball targets in frame | Far end cropped; 9/16 ball targets visible |
| Pool, backed-up end | Full cloth in frame | Full cloth, small far-edge margin | Far end substantially cropped |
| Pool, close side | Near corners outside horizontal view | Still too wide at near edge | Still too wide; tilt alone cannot solve it |
| Pool, backed-up side | Full cloth in frame | Full cloth in frame | Far edge cropped |

Lowering the lift at 32° improves close-table coverage, but still crops the bench
near edge and leaves more pool surface hidden behind rails. Height reduction alone
is not a universal solution. For whole-table perception, an end-on observation
position works better than standing close to a long side. A dedicated overview
pose can precede a closer manipulation/shot pose.

## What the metrics mean

`surface_in_image_pct`: uniformly sampled points on the rectangular tabletop/cloth
inside the pinhole image. `surface_visible_pct`: those points also passing a mesh
ray-occlusion test. Objects, rails and robot links can hide surface points even
when all four boundary corners are in the image. Pockets count as occluders of the
simplified underlying rectangle. These percentages are not depth accuracy.

`objects_unoccluded`: representative object points are in view and not blocked by
another geometry; a hit on the target's own geometry is accepted. This does not
claim the entire ball/object silhouette is visible. `surface_corners_in_image`
refers to the cloth/tabletop, not all outer rails or legs.

Measured-intrinsic rendering was independently checked by projecting a known 3D
point and measuring its rendered pixel location: **0.784 pixel error**, below the
1.5 pixel test limit. Every output image was decoded and inspected. Static bench
IK position errors are below 0.1 mm in these chosen illustrative poses, but
joint/path/payload/collision safety is not established by this study.

## Deliverables and reproduction

`runs/camera-view-study/summary.jpg` compares 32/42/55 degrees across six scenes.
Each `<scene>_comparison.jpg` also shows level, lower-lift, and legacy wide-FOV
cases. Native 1920 x 1080 images, external overview images, scene XML, complete
metrics and projection-check JSON are in the same directory.

```bash
cd ~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy/calibration
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../../../.venv/bin/python tools/camera/render_view_study.py
MUJOCO_GL=egl ../../../.venv/bin/python \
  tools/camera/verify_view_projection.py
```

Inputs are frozen in `config/camera/zed_view_study.json`. The first command builds
its GP80 scene if missing; it does not require a camera or network connection.

## Before calibration resumes

Decide the final mechanical mount angle, then confirm the real image sees the
board/manipulation region. Moving the camera relative to its mount frame invalidates
its calibrated transform. A manually adjustable mount needs repeatable indexed
poses with separate calibration, or a measured joint transform. Calibration will
solve camera-to-lift and board-to-right-flange; camera-to-base then follows measured
lift FK. The 30-pose motion is not yet validated or executed. An initial offline
approach candidate failed reachability and was not sent to the robot.

## Integrated ZED and bracket model

The third-person study now has an explicit camera-equipped URDF:
`urdf/maver_x4_gp80_zed.urdf`. It combines the ZED body with the short triangular v5 STL from
`~/artifacts/poolbot_camera_mounts/zed_short_triangular_v5_15deg/`. The bracket
STL is in millimetres, with explicit 0.001 URDF mesh scale; the camera body is in
metres. Its shelf is 65 mm above the bracket base, 100 mm lower than v3, and its
pitch is 15 degrees. The matching `proposed_camera_transform.json` is applied;
mesh inertia is recomputed assuming uniform density 1250 kg/m^3.
`zed_bracket_mount` attaches to `maver_x4_asm_URDF_link_1`, so the camera follows
the lift. The left/right optical frames and 120 mm stereo baseline are retained.
The view-only descriptor is `config/control/robot_models/maver-x4-gp80-zed-view-v1.json`;
existing hardware profiles remain bound to their original model.

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../../../.venv/bin/python tools/camera/render_view_study.py \
  --mounted --output runs/camera-view-study-short-v5
```

This produces six `<scene>_mounted_pair.jpg` sheets, each containing an external
view of the physical camera/bracket, the matching 1080p camera image, and a mount
close-up. Cyan lines bound the measured left-image field of view; orange is the
optical axis. The finite cone end is an illustrative section, not a maximum depth
range. Both the render and the frustum derive their pose from the URDF left optical
frame. This mode renders **only the short v5 STL's nominal 15-degree pitch**; the
42/55-degree comparison images above remain hypothetical orientations, not this
fixed bracket's geometry.

The ray-visibility check excludes only the camera housing body: MuJoCo's convex
mesh ray hull encloses the virtual pinhole. The mount and all robot links remain
occluders, and all meshes remain visible in rendered images. Mount-pose assumptions
are unchanged; integrating an STL does not measure the installed physical pose.

Validation: three regression checks cover unchanged robot joints/hand FK, stereo
baseline, camera-following-lift behavior, agreement of Pinocchio and MuJoCo optical
frames, and exact projection of frustum corners to image boundaries. The independent
render/pinhole pixel check also remains applicable.

The added geometry check pins the v5 mesh dimensions, unit scale, 65 mm shelf
datum and 15-degree transform, preventing accidental reuse of the tall bracket.
For this short mount at the measured lift height, the close bench retains only
2 of 4 object targets in frame. Backed-up end/side pool views retain all four
cloth corners and 16 ball targets. Lower mounting height alone does not compensate
for the shallower tilt; these remain nominal simulations pending calibration.

## Short support, angle-only comparison (current study)

The latest study holds the **65 mm shelf midplane datum**, base attachment,
65 x 60 x 6 mm base, 25 mm hole grid, two 4 mm ribs at y=±6.5 mm, and lift/arm
poses fixed. It compares 15, 32, 42 and 55 degrees down. The upper edges of the
rear wall and ribs meet each new shelf plane. Consequently the outer bounding-box
height changes with tilt; the shelf pivot stays at 65 mm. The camera-link origin
uses the same 3 mm shelf-normal offset, and the optical frame follows the camera.

`cad/build_short_zed_variants.py` derives the three comparison STLs from the saved
v5 short design. It checks connected/watertight geometry, open bores and the
specified screw-head clearances. The original 15-degree artifact is retained
byte-for-byte. All four simulation URDFs have consistent camera poses and inertia
estimates. These are alternative fixed-angle supports, not a modeled adjustable joint.

```bash
# Generate geometry with the CAD environment:
../PoolBot/.venv-cad/bin/python cad/build_short_zed_variants.py
# Render each angle, all six scenes:
for angle in 15 32 42 55; do
  MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ../../../.venv/bin/python tools/camera/render_view_study.py \
    --mounted --mount-angle "$angle" \
    --output "runs/camera-short-tilt-study/angle_$angle"
done
../../../.venv/bin/python tools/camera/assemble_short_tilt_study.py
```

Outputs in `runs/camera-short-tilt-study/` include the four-angle mount close-up,
`short_support_tilt_summary.jpg`, six large `<scene>_tilt_comparison.jpg` sheets
with third-person/frustum/first-person views, and the numeric comparison report.
All 24 cases use identical joint positions for each scene across angles.

| Tilt | Bench grasp targets | Close-end pool cloth corners / ball targets | Backed-up end cloth corners |
|---|---|---|---|
| 15° | 2/4 | 2/4 corners, 15/16 targets | 4/4 |
| 32° | 3/4 | 4/4 corners, 16/16 targets | 4/4 |
| 42° | 4/4 | 4/4 corners, 16/16 targets | 2/4 |
| 55° | 4/4; all bench corners in frame | 2/4 corners, 7/16 targets | 2/4 |

At 55°, part of the mount enters the lower-right camera image. At 42°, the close
bench's nearest edge still falls outside the image, even though all four object
target points are visible. Thus 42° is a compromise for the illustrated close-end
observation/manipulation cases; 32° covers more pool observation positions, while
55° prioritizes near-table coverage. These conclusions use nominal transforms and
do not establish physical extrinsics, trajectory safety or sensing quality.

Nine regression checks passed: four-angle FK consistency and lift following,
projection boundaries, original v5 units/dimensions, and the unchanged shelf datum
and base attachment across new angles. No robot motion was sent.

## Selected 40-degree print variant

The requested 40-degree STL is `urdf/meshes/zed/zed_short_triangular_40deg_mm.stl`.
It retains the 65 mm shelf datum and short ribbed structure. Overall print bounds
are 65 x 60 x 88.19 mm. Import in millimetres at 100% scale.
Generate it with `cad/build_short_zed_variants.py --angles 40` in the CAD environment.
The matching view-only URDF is `urdf/maver_x4_gp80_zed_40deg.urdf`, and rendering
accepts `--mounted --mount-angle 40`. This does not change hardware calibration.
