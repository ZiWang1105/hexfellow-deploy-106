# Right-wrist RGB and measured calibration geometry

The default native agent launch now supplies three images per observation:
head RGB, registered head depth visualization, and separately labeled wrist RGB.
Head intrinsics, head optical-to-base FK, measured left/right TCP poses and
physical proprioception remain. The action schema is unchanged: paired TCP
poses, binary grippers, and navigation XY/yaw.

The right D435i now uses the published candidate extrinsics at the user's request
(2026-10-06). The message includes `modality=rgb_only`,
`extrinsics_available=true`, original host receipt time, head time offset, and
allowlisted `geometry`. Geometry supplies active-profile K, fixed
`T_parent_from_optical`, dynamic `camera_T_base_from_optical`, and both TCP poses
at the matched wrist encoder time. The optical-to-base transform is
`FK_base_from_maver_x4_asm_mount2(q_measured) @ T_parent_from_optical`.
The fixed wrist frame is not a TCP frame. Translations are meters.
`geometry.right_tcp_T_from_wrist_optical` additionally maps camera optical points
into the configured right TCP frame. It includes the actual URDF/assembly tool
offset and is constant across arm/lift motion. The prompt uses it for gripper-relative
alignment while retaining base-frame output targets. Its composition with the
matched right TCP pose must equal the dynamic camera-to-base transform. This
adds a convenient coordinate representation, not new calibration accuracy or depth.

The candidate status, source SHA-256, and held-out RMS (11.3824 pixels at
1280 x 720) remain visible. This is not independently validated physical accuracy.
The policy prompt uses small wrist-guided alignment corrections and reobserves;
RGB alone cannot determine depth along a pixel ray. Head depth remains the range
source, never a depth map for wrist pixels. No wrist depth or contact truth is
invented. Board pose, calibration corners and datasets are excluded from inputs.

## Configuration and lifecycle

`configs/simulation.yaml` is the shared runtime configuration. Its
`wrist_camera` block is activated only for the native backend:

```yaml
wrist_camera:
  enabled: true
  serial: "243222074562"
  calibration: third_party/hexfellow-deploy/calibration/config/camera/realsense_d435i_right_candidate_20261006.json
  width: 640
  height: 480
  fps: 30
```

The explicit serial prevents accidentally selecting the uncalibrated left D405.
Calibration paths resolve relative to the parent workspace. The stream queries
its own SDK K at 640 x 480; it does not scale the 1280 x 720 calibration K across
aspect ratios. Nonzero distortion is rejected pending explicit rectification.
RGB and matched encoder/FK metadata publish atomically in DA shared memory.
A separate feedback sampler reads the existing encoder topic; camera work does
not block the controller. Frames without a match within the configured encoder
skew (at most 80 ms) are dropped. The source calibration file is archived at the
run root as `wrist_calibration.json`; `wrist_camera_health.json` records the active
K, identity, calibration provenance, unmatched frames and publication counts.
Install the parent `realsense` extra or its pinned `pyrealsense2==2.58.4.10922`
package in the runner environment; it was installed on this desktop for the
preceding camera preview.

The existing real-agent command activates this input automatically. The
dashboard at port 8765 includes a labeled wrist preview and candidate-calibration status. It returns HTTP 503
and marks the view stale when no recent wrist image is available. Policy waits
for a wrist frame at most 0.5 seconds old and no more than 0.15 seconds from the
head observation's timestamp. It does not silently drop a configured wrist view.
Those bounds limit host receipt skew; the cameras are not hardware synchronized.
Controller timing and output contracts are unchanged.

`--set wrist_camera.enabled=false` explicitly selects the older head-only input.
The legacy viewer's disabled `cameras.wrist` configuration is separate; enabling
this RGB-only policy component does not claim a secondary RGB-D camera.
`--set wrist_camera.calibration=null` explicitly selects uncalibrated RGB context;
only that mode permits serial auto-selection with exactly one connected device. The primary head remains the configured ZED.

Simulation does not open this physical camera and does not invent wrist mounting
geometry. Its head-only runs remain labeled accordingly. The common policy
serializer accepts the same optional RGB-only observation schema on either
backend when that sensor is available.

## Calibrated integration validation

Offline tests cover real URDF FK under right-arm/lift motion, projection direction,
strict metadata filtering, source identity, active-profile K, encoder matching,
atomic RGB/geometry publication, and the actual model-request JSON with a stub API.
The request auditor verifies the archived calibration hash and recomputes wrist
camera/TCP FK from the matched physical joints. A deliberately corrupted dynamic
transform is rejected even when both request and turn records are changed.
A fake SDK capture verifies unmatched frames are dropped and resources close.
These checks do not establish live fine-adjustment or grasp success. No hardware
or remote model call was used for this integration validation.

## Historical uncalibrated validation (2026-10-05)

196 targeted tests passed: policy boundary, policy process, camera viewer, and
the complete deployment suite. Tests inspect the actual outgoing request body
with a stub API: exactly three images, unchanged head metadata/action schema,
pixel-identical archived wrist PNG, and no wrist depth/calibration fields.
Offline request audit checks the auxiliary image/history against the archive.
Tests cover stale/future/missing wrist frames, semaphore contention, native-only
YAML wiring and a dashboard 503 response for stale wrist pixels.

The actual RGB-only component captured the connected D435i through the real DA
RGB topic. 100 distinct frames were observed at 29.98 FPS; maximum consumer age
was 5.66 ms, 101 frames were published, and there were no publication misses or
capture errors. It exited normally and released the camera. Robot SDK and ZED
imports were blocked in that probe. There were no model calls or robot commands.
Raw evidence: parent `runs/wrist_rgb_policy_check`; compact report in
`evidence/20261005_wrist_rgb/probe.json`.

An initially broader test invocation also encountered an existing base-pulse
duration assertion (1.03 s versus 1.0 s), reproduced after loading the unchanged
HEAD contract module. That controller behavior is outside this sensor change.
The first invocation lacked a headless OpenGL setting; rendering tests require
`MUJOCO_GL=egl`. This validation is not a new agent pickup or simultaneous
head/wrist physical-control trial.
