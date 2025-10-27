#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/training.sdxl-qlora.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/configs/accelerate/default_config.yaml}"

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found at ${ACCELERATE_CONFIG}."
  echo "Run 'accelerate config' and save to configs/accelerate/default_config.yaml."
  exit 1
fi

source "${PROJECT_ROOT}/.venv/drawing-agent/bin/activate"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m src.tuning.cli \
  train \
  --config "${CONFIG_PATH}"
