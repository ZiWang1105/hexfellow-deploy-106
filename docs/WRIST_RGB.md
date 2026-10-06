# Uncalibrated wrist RGB policy context

The default native agent launch now supplies three images per observation:
head RGB, registered head depth visualization, and separately labeled wrist RGB.
Head intrinsics, head optical-to-base FK, measured left/right TCP poses and
physical proprioception remain. The action schema is unchanged: paired TCP
poses, binary grippers, and navigation XY/yaw.

The wrist camera is **not geometrically calibrated**. Its message contains only
RGB pixels and original host receipt time. The model sees `modality=rgb_only`,
`extrinsics_available=false`, and `head_time_offset_s`. No wrist depth, intrinsic
matrix or camera-to-base matrix is provided. The system prompt explicitly forbids
applying the head camera's geometry to wrist pixels. Wrist images can inform
visibility, occlusion, apparent finger alignment and grasp progress; they are not
contact measurements or a calibrated 3D target source.

## Configuration and lifecycle

`configs/simulation.yaml` is the shared runtime configuration. Its
`wrist_camera` block is activated only for the native backend:

```yaml
wrist_camera:
  enabled: true
  serial: null
  width: 640
  height: 480
  fps: 30
```

A null serial requires exactly one connected RealSense device. For multiple
devices, append `--set wrist_camera.serial=243222074562` using the intended
camera's serial. The connected desktop device identifies itself as a D435i.
The capture process enables only its color stream and publishes a DA numeric
RGB topic, independently of robot feedback, IK, head capture and inference.
Install the parent `realsense` extra or its pinned `pyrealsense2==2.58.4.10922`
package in the runner environment; it was installed on this desktop for the
preceding camera preview.

The existing real-agent command activates this input automatically. The
dashboard at port 8765 includes a labeled wrist preview. It returns HTTP 503
and marks the view stale when no recent wrist image is available. Policy waits
for a wrist frame at most 0.5 seconds old and no more than 0.15 seconds from the
head observation's timestamp. It does not silently drop a configured wrist view.
Those bounds limit host receipt skew; the cameras are not hardware synchronized.
Controller timing and output contracts are unchanged.

`--set wrist_camera.enabled=false` explicitly selects the older head-only input.
The legacy viewer's disabled `cameras.wrist` configuration is separate; enabling
this new RGB-only policy component does not claim a calibrated secondary RGB-D
camera. The primary head remains the configured ZED.

Simulation does not open this physical camera and does not invent wrist mounting
geometry. Its head-only runs remain labeled accordingly. The common policy
serializer accepts the same optional RGB-only observation schema on either
backend when that sensor is available.

## Validation

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
