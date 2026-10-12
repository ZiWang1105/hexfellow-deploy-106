#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${HEXFELLOW_CALIBRATION_PYTHON:-$deploy_root/.venv/bin/python}" \
  "$deploy_root/calibration/cli.py" "$@"
