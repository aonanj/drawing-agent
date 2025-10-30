
import os
from pathlib import Path

def get_basedir():
    """
    Returns the base directory for the project.
    If the GOOGLE_DRIVE_PATH environment variable is set, it will be used as the base directory.
    Otherwise, the current working directory will be used.
    """
    return Path(os.environ.get("GOOGLE_DRIVE_PATH", "/content/drive/MyDrive/colab/drawing-agent"))

BASE_DIR = get_basedir()
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCRIPTS_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "src"

# Dataset-specific paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = DATA_DIR / "figures"
CLAIMS_DIR = DATA_DIR / "claims"

# Figures paths
FIGURES_RAW_DIR = FIGURES_DIR / "raw"
FIGURES_PROCESSED_DIR = FIGURES_DIR / "processed"
FIGURES_CACHE_DIR = FIGURES_DIR / "cache"

# Claims paths
CLAIMS_RAW_DIR = CLAIMS_DIR / "raw"
CLAIMS_PROCESSED_DIR = CLAIMS_DIR / "processed"
CLAIMS_CACHE_DIR = CLAIMS_DIR / "cache"

# Outputs
SDXL_QLORA_OUTPUT_DIR = OUTPUTS_DIR / "sdxl-qlora"
TEST_OUTPUT_DIR = OUTPUTS_DIR / "test"
