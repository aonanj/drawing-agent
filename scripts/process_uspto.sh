#!/usr/bin/env bash
set -euo pipefail

# If GOOGLE_DRIVE_PATH is set, use it as the project root.
# Otherwise, use the parent directory of the script's location.
PROJECT_ROOT="${GOOGLE_DRIVE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/figures.yaml}"

# Activate virtual environment
VENV_PATH="${PROJECT_ROOT}/.venv"
if [[ -d "${VENV_PATH}" ]]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Virtual environment not found at ${VENV_PATH}."
    echo "Please run 'scripts/setup_environment.sh' or ensure your environment is correctly set up."
    exit 1
fi

echo "========================================"
echo "  USPTO Patent Figure Processing"
echo "========================================"
echo "Project Root: ${PROJECT_ROOT}"
echo ""

# Run diagnostics first
echo "Step 1: Running diagnostics..."
echo "----------------------------------------"
python "${PROJECT_ROOT}/diagnose_data.py" --raw-dir "${PROJECT_ROOT}/data/raw"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Diagnostics failed. Please fix issues before processing."
    exit 1
fi

echo ""
echo "Step 2: Processing patents..."
echo "----------------------------------------"
echo "Config: ${CONFIG_PATH}"
echo "Raw data directory: ${PROJECT_ROOT}/data/raw"
echo ""

# Run the processing command
python -m src.tuning.cli process-figures --config "${CONFIG_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Processing Complete!"
    echo "========================================"
    echo ""
    echo "Dataset saved to: data/figures/processed/"
    echo ""
    echo "Next steps:"
    echo "  1. Review the dataset: data/figures/processed/hf_dataset"
    echo "  2. Check summary: data/figures/processed/dataset_summary.json"
    echo "  3. Run training: scripts/run_finetune.sh"
else
    echo ""
    echo "❌ Processing failed. Check the logs above for errors."
    exit 1
fi