# Standalone robot runtime

`hexfellow-deploy` owns robot control, hardware interfaces, sensor/action contracts,
robot-relative kinematics, MuJoCo construction/assets, gripper regulation and the
interactive TCP viewer. An agent application consumes these interfaces; it is not
needed to install or run the robot simulator.

## Install and launch

On Ubuntu, install `build-essential`, `libeigen3-dev`, and `uv`. Clone this repository normally (no submodules), then run:

```bash
./scripts/setup_sim.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/hexfellow-deploy \
  --interactive-ik --no-camera \
  --output "runs/interactive_ik_$(date +%Y%m%d_%H%M%S)"
```

Use a desktop terminal with DISPLAY. Double-click a red TCP sphere. Ctrl + right
mouse drag translates; Ctrl + left mouse drag rotates. Space/R requests HOLD and
resets the handles to measured TCPs. Q closes the viewer and shared runtime.
The viewer owns no physics stepping and never writes simulator joint state.

`--workspace` is optional and means the root for explicit configuration paths,
not a required agent checkout. Defaults come from this deployment checkout.
The example deployment is uncommissioned and does not authorize hardware.

For a headless shared MIT controller test:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/hexfellow-deploy \
  --no-camera --duration-s 8 --action tests/fixtures/paired_pose.json \
  --set interface.control.startup_carry=false \
  --output "runs/ik_$(date +%Y%m%d_%H%M%S)"
```

The action fixture is a paired target diagnostic, not a continuous trajectory
benchmark or a successful grasp. Control records, actuator receipts, resolved
configuration and provenance remain available under the selected output directory.

## Ownership and optional applications

- `control/`: paired TCP execution, owned IK adapter, startup, limits.
- `robot/`: measured-joint FK, assembly and owned resource resolution.
- `hardware/`: deployment parsing, native endpoint and camera capture adapters.
- `interfaces/`: measured sensor and action contracts, depth/camera geometry.
- `sim/`: MuJoCo scene construction, sensors, assets and recording.
- `skills/`: existing robot-relative primitive sequencing; no model/API client.
- `robotics/`: owned PinkIK core, MIT model, device transport and calibration algorithms.
- `runtime/`: owned process lifecycle, configuration and shared-memory communication.
- `robot_resources/`: robot URDF, meshes and static controller/camera configurations.

The application retains its model client, prompts, grounding, task catalog and
scoring. It explicitly passes `--agent-plugin module.name`; that module exports
`AGENT_COMPONENT` (an importable class path), and optional `load_catalog`,
`materialize_scene`, and `evaluate_run` hooks for catalog tasks. Alternatively,
`policy.component` can select a component for a direct prompt. No application is
imported by an IK-only run. Model credentials remain the application's concern.

Old application robot-module paths are compatibility aliases to the deployment
implementation. They preserve object identity and do not maintain another solver.
Robot-source SHA-256 values before extraction are recorded in
`ROBOT_CODE_PROVENANCE.json`, including preexisting working-tree edits.
Asset attribution and skill provenance files are preserved. Historical experiment
outputs and machine-local hardware reviews were not copied into deployment defaults.

The extraction preserves the current controller, MIT commands, feedback and
ownership checks, fixed-lift startup/braking changes and gripper regulation. It
does not retune speed, alter torque limits, promote calibrations, or qualify new
physical behavior. Existing hardware configuration/release bindings remain local;
source relocation does not regenerate approvals.

## Validation

The deployment tests use this repository's own configurations and assets. Native
wire-format tests require the optional SDK but open no devices:

```bash
uv pip install --python .venv/bin/python -e '.[hardware]'
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q
```

Camera calibration tests use a separate process to keep their tool import setup separate:

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest calibration/tests -q
```

Standalone validation also copies the worktree into an unrelated directory,
creates a fresh virtual environment, builds its native extension, and blocks all
imports of the agent package and both removed packages while exercising simulation and real GUI input.
Passing these checks establishes repository independence and modeled behavior,
not hardware reliability or physical safety qualification.

## Dependency removal (2026-10-14)

The former `policy_sim` and `deploy_anything` packages are not dependencies.
The two Git submodules have been removed. The native collision/constraint kernel
is built by this package as `_hexfellow_collision_kernel`. The Python modules
are owned under `hexfellow_deploy.robotics` and `hexfellow_deploy.runtime`;
calibration uses the same owned code. Scientific dependencies such as MuJoCo,
Pinocchio, Pink, and the QP solver remain ordinary versioned Python packages.

Historical names in provenance and original source comments retain attribution;
see [NOTICE.md](../NOTICE.md). They do not trigger external imports or downloads.
Existing local hardware source bindings will differ after the code relocation.
This extraction does not regenerate approvals or re-enable device motion.

The October 14 isolated-install, import-blocking, solver, calibration and native
mouse-drag results are recorded in [owned-runtime validation](validation/owned_runtime_20261014.json).
The migration leaves three preexisting application navigation-test assertions
unresolved; the robot deployment and calibration suites pass.
