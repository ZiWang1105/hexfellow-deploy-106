#!/usr/bin/env bash
# Reproducible source-checkout install. No application checkout is required.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v c++ >/dev/null 2>&1 && [ -z "${CXX:-}" ]; then
  echo "Install build prerequisites first: sudo apt-get install build-essential libeigen3-dev" >&2
  exit 1
fi
git submodule update --init --recursive
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e '.[sim,dev]'
uv pip install --python .venv/bin/python 'setuptools>=68,<82' wheel 'pybind11>=2.11,<4'
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e third_party/UMI_Hardware
# DA's optional backends require NumPy 2; its manager/topics run with the robot's NumPy 1.x.
uv pip install --python .venv/bin/python log-throttling
uv pip install --python .venv/bin/python --no-deps -e third_party/deploy-anything
