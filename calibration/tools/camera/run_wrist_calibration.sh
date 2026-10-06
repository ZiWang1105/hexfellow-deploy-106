#!/usr/bin/env bash
set -euo pipefail
camera_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$camera_root/tools/camera/run_calibration.sh" \
  --wrist-plan "$camera_root/config/camera/realsense_d435i_right_fixed33_outside_plan.json" "$@"
