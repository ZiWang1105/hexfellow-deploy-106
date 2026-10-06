#!/usr/bin/env bash
set -euo pipefail
calibration_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$calibration_root/tools/camera/run_calibration.sh" \
  --validation-plan "$calibration_root/config/camera/left_hand_validation.json" \
  --port 8773 "$@"
