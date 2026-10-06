#!/usr/bin/env bash
set -euo pipefail
calibration_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="$(cd -- "$calibration_root/../../.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
exec "${HEXFELLOW_CALIBRATION_PYTHON:-$workspace/.venv/bin/python}" -u \
  "$calibration_root/tools/camera/calibrate_robot.py" "$@"
