#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd -- "$deploy_root/../.." && pwd)"
exec "${HEXFELLOW_CALIBRATION_PYTHON:-$workspace/.venv/bin/python}" \
  "$deploy_root/calibration/cli.py" "$@"
