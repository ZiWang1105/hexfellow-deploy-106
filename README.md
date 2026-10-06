# Hexfellow Deploy

Deploy Anything (DA) YAML, process lifecycle and named shared-memory topics for
Hexfellow, with the existing UMI PinkIK solver and RGB-D agent interface.

**Current scope: runnable MuJoCo deployment and IK qualification. The native
MIT hardware adapter is not implemented in this repository yet. This is not a
replacement real-robot launch command.** The old gateway is not imported by the
new runtime. Selecting a hardware backend fails before opening any device.

DA is a Git submodule pinned to `6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`.
Robot assets, sensor contracts, policy and PinkIK adapters currently come from
the enclosing `agentic-mobile-manipulator` checkout via `--workspace`. This
workspace dependency is explicit; the repository is not yet a standalone robot
distribution. See [architecture and remaining work](docs/ARCHITECTURE.md) and
[measured results](docs/VALIDATION.md).

## Installation in the existing workspace

Use the established parent Python 3.10 environment with MuJoCo, Pinocchio and the
UMI native extension already installed. From this repository:

```bash
git submodule update --init --recursive
uv pip install --python ../../.venv/bin/python \
  'hydra-core==1.3.7' 'posix_ipc==1.3.2' 'rich==15.0.0' log-throttling
uv pip install --python ../../.venv/bin/python --no-deps -e third_party/deploy-anything -e .
```

The tested environment uses NumPy 1.26.4 because the parent Pinocchio environment
requires NumPy below 2. DA's package metadata requests NumPy 2; `--no-deps` is an
intentional compatibility exception for its tested shared-memory/manager subset.
Do not upgrade the robot environment to NumPy 2 to satisfy that metadata. ROS,
YAM, CAN and DA's optional learning backends are not used here.

## Run the simulator

From `agentic-mobile-manipulator`, run an actual asynchronous agent:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hexfellow_deploy.cli \
  --workspace . \
  --prompt 'Pick up the apple and hold it above the table.' \
  --output "third_party/hexfellow-deploy/runs/agent_$(date +%Y%m%d_%H%M%S)"
```

The terminal prints the clickable dashboard URL, normally
<http://127.0.0.1:8765/>. It displays RGB, current control state, actuator
acceptance, the latest agent action, model-call state and errors. The last action
remains visible during the next inference. `decisions.jsonl` preserves transitions.
API requests, images, responses and reported costs are archived under `policy/`.
The API client alone reads the existing external key file.

The default model is Astra with medium effort, configured in
`configs/simulation.yaml`. There is no default run duration, call count or API
spending cap. Ctrl-C closes the run. `--duration-s` is an optional whole-run
diagnostic duration, counted after control startup, not a per-call timeout.

For a reproducible controller diagnostic without API calls:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m hexfellow_deploy.cli \
  --workspace . --duration-s 26 --no-camera \
  --action third_party/hexfellow-deploy/tests/fixtures/paired_pose.json \
  --set interface.control.inject_every=10 \
  --set interface.control.inject_delay_s=0.03 \
  --set interface.control.submit_delay_s=0.03 \
  --output "third_party/hexfellow-deploy/runs/delay_$(date +%Y%m%d_%H%M%S)"
```

Use `--set interface.dashboard.port=8875` to avoid another viewer's port.
`--set interface.control.startup_carry=true` enables the existing task-independent
elbow carrying preparation. `--set` accepts OmegaConf YAML overrides; topic
references resolve through DA's topic registry. All components run asynchronously.

## Verification

From this repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../../.venv/bin/python -m pytest tests -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../../.venv/bin/python scripts/qualify_ik.py \
  --workspace ../.. --output runs/ik_qualification_new
../../.venv/bin/python scripts/summarize_run.py runs/delay_example
```

Run outputs include `control.jsonl`, `control_final.json`, configuration,
invocation and code/dependency provenance. Qualification separates desired TCP
error of the reference from desired TCP error of measured encoders. No simulator
pass establishes physical MIT gains, firmware ownership, transport behavior,
gripper calibration, stopping, collision clearance or pickup success.
