# PoolBot controller environment

The PoolBot deployment controller uses Python 3.11. Its Isaac Sim 6 plant stays
in PoolBot's existing Python 3.12 `.venv-sim6` and communicates over ZMQ. Do not
install the controller's NumPy/Pinocchio dependencies into the Isaac environment.
These installation steps do not connect devices or start a simulator.

This repository now owns both transitive source pins:

- `third_party/deploy-anything`: `6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`.
- `third_party/policy-sim`: `9398b61972bf983c7ab5609591b7eb4c95c6357d`.

The policy-sim pin matches the shared agentic-mobile-manipulator runtime. PoolBot
previously used `32d373cb584783046349f53f0343f518cad56d6c`. UMI/PinkIK remains
an internal transitive dependency owned by hexfellow-deploy; it was not removed
or replaced by a different solver. Do not import the sibling workspace checkout.

PoolBot supplies its own robot description, meshes, camera/cue assembly, and
the unchanged `x7-gp80-highspeed-fixed-base-v1` tuning. The portable timed executor
keeps its explicit planar 18-coordinate internal model and 15-encoder wire map.
The shared manipulator desktop uses a fixed 15-coordinate model and additional
application lift ceilings; those deployment settings are not PoolBot settings.
This integration shares reference ownership, receipts and actuator contracts,
with a portable timed executor and an Isaac/ZMQ adapter. It does not claim that
PoolBot runs the identical DA Manager, static PoseExecutor or hardware process
graph. Changing source ownership does not substitute assets or increase limits.

## New environment

From the PoolBot root, with this repository at `third_party/hexfellow-deploy`:

```bash
git submodule update --init --recursive third_party/hexfellow-deploy
uv venv --python 3.11 .venv-deploy
uv pip install --python .venv-deploy/bin/python \
  -r third_party/hexfellow-deploy/third_party/policy-sim/requirements-simulation.lock \
  -r requirements/web.lock cmeel-eigen==3.4.1
uv pip install --python .venv-deploy/bin/python -e '.[web,dev]' \
  hydra-core==1.3.7 posix_ipc==1.3.2 rich==15.0.0 log-throttling==0.0.3
uv pip install --python .venv-deploy/bin/python --no-deps \
  -e third_party/hexfellow-deploy/third_party/deploy-anything \
  -e third_party/hexfellow-deploy
```

The `--no-deps` exception is intentional: DA declares NumPy 2 while the tested
manager/shared-memory subset works with the controller's NumPy 1.26.4. Never
upgrade NumPy merely to silence this metadata conflict. `uv pip check` must show
only that documented exception. Do not add DA aggregate `robot`, `vision`, or
`dev` extras. Hexfellow's optional validation extra also requests Matplotlib
below 3.10, whereas this policy lock uses 3.10.8; it is not installed here.

## Native controller extension

For a fresh build, install a C++17 compiler and use the pinned source normally:

```bash
uv pip install --python .venv-deploy/bin/python --no-deps --no-build-isolation \
  -e third_party/hexfellow-deploy/third_party/policy-sim
```

When a separate environment still contains a qualified CPython 3.11 artifact,
the bounded reuse path is (replace the example environment path):

```bash
uv pip install --python .venv-deploy/bin/python patchelf==0.17.2.4
.venv-deploy/bin/python third_party/hexfellow-deploy/scripts/bootstrap_policy_native.py \
  --reuse-from-python /path/to/qualified-environment/bin/python
```

The helper verifies clean exact source revisions, relevant source hashes,
CPython/native ABI, dependency versions and every existing CMeel shared-library
hash. It copies the native module into the new environment and changes its RPATH
to `$ORIGIN/cmeel.prefix/lib`, so it does not depend on the old environment at
runtime. An explicit `.pth` file selects the nested pinned Python source. This
reuse path does not register a separate `policy-sim` distribution; the deployment
source loader owns that dependency and checks its source identity.

`.venv-deploy/policy_native_provenance.json` records both binary hashes, source
and dependency identities and resolved native libraries. Reuse verifies the
existing artifact's consistency; it does **not** claim an independently reproduced
native build. Preserve this distinction in qualification reports.

On the migration host, the binary is already installed and qualified in
`.venv-deploy`; the former `.venv-umi` source checkout is no longer required.
After upgrading from the original PoolBot pin to the shared runtime pin, run:

```bash
.venv-deploy/bin/python third_party/hexfellow-deploy/scripts/bootstrap_policy_native.py \
  --revalidate-installed
```

This verifies identical native build inputs at both Git revisions, the installed
binary hash, ABI, dependency versions and library bytes. It preserves the old
manifest and records the original artifact revision separately from the new
Python source revision. The new source adds fixed-base asynchronous feedback
checks and measured-lift braking feasibility; profiles and native build inputs
are unchanged. Those Python behavior changes require controller regression and
simulator qualification even though the native binary does not need rebuilding.

No `.so`, virtual environment or generated provenance is committed. The nested
source checkout is not edited by the reuse helper.

## Verification and recording

Run the pinned upstream numerical/native regressions without a plant:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-deploy/bin/python -m pytest \
  third_party/hexfellow-deploy/third_party/policy-sim/tests/control/test_target_motion.py \
  third_party/hexfellow-deploy/third_party/policy-sim/tests/control/test_fast_ik_native.py \
  third_party/hexfellow-deploy/third_party/policy-sim/tests/control/test_measured_lift_boundary.py \
  third_party/hexfellow-deploy/third_party/policy-sim/tests/control/test_fixed_feedback_progress.py -q
uv pip check --python .venv-deploy/bin/python
```

Verify `policy_sim` resolves inside this repository and the native module resolves
inside `.venv-deploy`, with ABI 3. PoolBot's UI browser automation can continue
using its existing `.venv` Playwright/Chromium installation. `.venv-deploy` has
PyAV 16.1.0 and ImageIO's bundled FFmpeg for recording. No browser or hardware SDK
is required in the controller environment. Preview frame rate alone does not
establish simulation-time playback; retain exposure timestamps when making
lifecycle videos.
