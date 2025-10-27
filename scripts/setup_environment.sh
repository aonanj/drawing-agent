#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-drawing-agent}"

python3.12 -m venv "${PROJECT_ROOT}/.venv/${ENV_NAME}"
source "${PROJECT_ROOT}/.venv/${ENV_NAME}/bin/activate"
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "Environment '${ENV_NAME}' created and dependencies installed."
