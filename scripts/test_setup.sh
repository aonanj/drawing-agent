#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Drawing Agent - Setup Validation"
echo "========================================"
echo ""

# Check 1: Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Check 2: Virtual environment
echo -n "Checking virtual environment... "
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    echo -e "${GREEN}✓${NC} Found"
else
    echo -e "${YELLOW}!${NC} Not found. Run: bash scripts/setup_environment.sh"
fi

# Check 3: Activate environment for remaining checks
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"

    # Check 4: Required packages
    echo -n "Checking required packages... "
    MISSING_PACKAGES=()
    
    for pkg in torch diffusers transformers peft datasets accelerate opencv-python pillow typer; do
        if ! python -c "import $pkg" 2>/dev/null; then
            MISSING_PACKAGES+=($pkg)
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All installed"
    else
        echo -e "${RED}✗${NC} Missing: ${MISSING_PACKAGES[*]}"
        echo "   Run: pip install -r requirements.txt"
    fi
    
    # Check 5: CUDA availability
    echo -n "Checking CUDA/GPU... "
    CUDA_CHECK=$(python -c "import torch; print('available' if torch.cuda.is_available() else 'not available')" 2>/dev/null)
    if [ "$CUDA_CHECK" = "available" ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "${GREEN}✓${NC} $GPU_COUNT GPU(s) available - $GPU_NAME"
    else
        echo -e "${YELLOW}!${NC} No CUDA GPUs detected (CPU-only mode)"
    fi
fi

# Check 6: Directory structure
echo -n "Checking directory structure... "
MISSING_DIRS=()
for dir in data/raw configs scripts src/tuning docs; do
    if [ ! -d "${PROJECT_ROOT}/$dir" ]; then
        MISSING_DIRS+=($dir)
    fi
done

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All directories present"
else
    echo -e "${RED}✗${NC} Missing: ${MISSING_DIRS[*]}"
fi

# Check 7: Configuration files
echo -n "Checking configuration files... "
MISSING_CONFIGS=()
for config in configs/figures.yaml configs/training.sdxl-qlora.yaml configs/dataset.yaml; do
    if [ ! -f "${PROJECT_ROOT}/$config" ]; then
        MISSING_CONFIGS+=($config)
    fi
done

if [ ${#MISSING_CONFIGS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All configs present"
else
    echo -e "${RED}✗${NC} Missing: ${MISSING_CONFIGS[*]}"
fi

# Check 8: Raw data
echo -n "Checking raw data... "
if [ -d "${PROJECT_ROOT}/data/raw" ]; then
    BULK_FOLDERS=$(find "${PROJECT_ROOT}/data/raw" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ $BULK_FOLDERS -gt 0 ]; then
        ZIP_COUNT=$(find "${PROJECT_ROOT}/data/raw" -name "*.ZIP" | wc -l)
        echo -e "${GREEN}✓${NC} Found $BULK_FOLDERS bulk folder(s) with $ZIP_COUNT zip files"
    else
        echo -e "${YELLOW}!${NC} No bulk download folders found in data/raw/"
        echo "   Download USPTO data and place in data/raw/"
    fi
else
    echo -e "${YELLOW}!${NC} data/raw directory not found"
fi

# Check 9: Processed data
echo -n "Checking processed data... "
if [ -d "${PROJECT_ROOT}/data/figures/processed/hf_dataset" ]; then
    TRAIN_COUNT=$(python -c "from datasets import load_from_disk; ds = load_from_disk('data/figures/processed/hf_dataset'); print(len(ds['train']))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} Dataset exists with $TRAIN_COUNT training samples"
else
    echo -e "${YELLOW}!${NC} No processed dataset found"
    echo "   Run: bash scripts/process_uspto.sh"
fi

# Check 10: Accelerate config
echo -n "Checking accelerate config... "
if [ -f "${PROJECT_ROOT}/configs/accelerate/default_config.yaml" ]; then
    echo -e "${GREEN}✓${NC} Found"
else
    echo -e "${YELLOW}!${NC} Not found. Run: accelerate config"
    echo "   Save to: configs/accelerate/default_config.yaml"
fi

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

# Determine overall status
if [ -d "${PROJECT_ROOT}/.venv" ] && \
   [ -d "${PROJECT_ROOT}/data/raw" ] && \
   [ -f "${PROJECT_ROOT}/configs/figures.yaml" ]; then
    echo -e "${GREEN}✓${NC} Basic setup complete"
    
    if [ -d "${PROJECT_ROOT}/data/figures/processed/hf_dataset" ]; then
        echo -e "${GREEN}✓${NC} Data processed and ready for training"
        echo ""
        echo "Next step: bash scripts/run_finetune.sh"
    else
        echo -e "${YELLOW}!${NC} Ready to process data"
        echo ""
        echo "Next step: bash scripts/process_uspto.sh"
    fi
else
    echo -e "${RED}✗${NC} Setup incomplete"
    echo ""
    echo "Follow QUICKSTART.md for setup instructions"
fi

echo ""