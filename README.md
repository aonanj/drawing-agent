# drawing-agent: SDXL QLoRA Fine-Tuning Workspace

This project contains the workspace scaffolding for fine-tuning a Stable Diffusion XL model using QLoRA to generate patent-ready drawings from claim text.

## Repository Layout

- `configs/` — YAML configuration files for datasets, training, and inference.
- `data/` — Placeholders for input claims datasets and generated artifacts.
- `docs/` — Blueprint documentation and research notes.
- `notebooks/` — Optional exploratory data analysis and prototype notebooks.
- `scripts/` — Shell utilities for environment setup and training workflows.
- `src/tuning/` — Python package with data preparation and training entrypoints.

## Quick Start

1. Create and activate a Python 3.10+ environment.
2. Install project dependencies: `pip install -r requirements.txt`.
3. Copy or generate your claims dataset into `data/claims/`.
4. Update configuration files under `configs/` to match the dataset and hardware.
5. Run `scripts/run_finetune.sh` to launch QLoRA training with Accelerate.

See `docs/blueprint.md` for detailed architecture, data flow, and next steps.

