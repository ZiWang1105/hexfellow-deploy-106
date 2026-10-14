#!/usr/bin/env bash
# Build the deployment-owned controller and native kernel. No submodules required.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v c++ >/dev/null 2>&1 && [ -z "${CXX:-}" ]; then
  echo "Install build prerequisites first: sudo apt-get install build-essential libeigen3-dev" >&2
  exit 1
fi
uv python install 3.10
uv venv --python 3.10 .venv
# Install native build dependencies into the same environment as the final extension.
uv pip install --python .venv/bin/python 'setuptools>=68,<82' wheel 'pybind11>=2.11,<4' 'hpp-fcl==2.4.4' 'numpy>=1.26,<2'
uv pip install --python .venv/bin/python --no-build-isolation -e '.[sim,dev]'
