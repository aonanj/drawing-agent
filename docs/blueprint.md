# SDXL QLoRA Fine-Tuning Blueprint

## Goal

Fine-tune a Stable Diffusion XL base model using QLoRA so the model can produce patent-style drawings conditioned on a set of parsed patent claims and optional textual context.

## High-Level Phases

1. **Dataset Curation**
   - Parse patent claims into structured prompts.
   - Collect paired reference drawings when available.
   - Generate negative prompts describing artifacts to avoid.
   - Perform text and image quality checks.
2. **Feature Engineering**
   - Convert claims into machine-readable prompt templates.
   - Extract metadata such as CPC codes, claim dependencies, and figure references.
   - Tokenize prompts with SDXL tokenizer; store as HF datasets.
3. **Training & Evaluation**
   - Apply QLoRA adapters on SDXL base model to reduce GPU memory.
   - Use Accelerate + bitsandbytes for distributed mixed-precision training.
   - Evaluate with CLIP score, structural similarity (SSIM), and qualitative review.
4. **Deployment**
   - Package trained adapters.
   - Expose inference through scripts or APIs.

## Repository Structure

```
configs/
  dataset.yaml            # Dataset structure, prompt templating, splits.
  training.sdxl-qlora.yaml # Training hyperparameters and hardware config.
  inference.yaml          # Optional inference defaults.
data/
  claims/                 # Raw and processed claim text files.
  reference/              # Supporting drawings or annotations.
docs/
  blueprint.md            # (this file) Architectural and process blueprint.
  research-notes.md       # Literature, experiments, references. (optional)
notebooks/
  exploration/            # Jupyter notebooks for EDA.
scripts/
  setup_environment.sh    # Create/activate virtualenv/conda, install deps.
  run_preprocess.sh       # Batch data prep and dataset creation.
  run_finetune.sh         # Launch Accelerate + training entrypoint.
src/
  tuning/
    __init__.py
    cli.py                # Typer CLI aggregator for dataset/train commands.
    dataset.py            # Dataset parsing/packing logic.
    preprocess.py         # Text/image cleaning routines.
    training.py           # QLoRA training loop.
    utils.py              # Shared helper functions.
```

## Data Blueprint

- **Inputs**
  - CSV/JSONL with fields: `claim_id`, `claim_text`, `dependent_on`, `category`, `figure_ids`.
  - Optional image paths in `reference_path`.
  - Negative prompt hints derived from patent domain heuristics.
- **Processing**
  - Validate claim language using regex + language detection.
  - Normalize terminology (e.g., "embodiment", "member").
  - Template prompts: `"Patent drawing of {category} illustrating {main_feature} ..."`
  - Store in Hugging Face `datasets.Dataset` with train/val/test splits.
  - Persist dataset config in `data/claims/dataset_config.json`.

## Training Blueprint

- **Base model**: `stabilityai/stable-diffusion-xl-base-1.0`.
- **LoRA backend**: QLoRA with 4-bit NF4 quantization via `bitsandbytes`.
- **Optimizer**: AdamW8bit, cosine LR schedule, warmup steps configurable.
- **Batching**: Gradient accumulation to simulate large batch sizes.
- **Tools**: `accelerate`, `transformers`, `diffusers`, `peft`, `datasets`.
- **Logging**: `wandb` or `tensorboard` optional.
- **Checkpointing**: Save LoRA adapters every N steps, final adapter + config.

## Evaluation Blueprint

- Generate validation images per claim subset.
- Compute automatic scores:
  - CLIP image-text similarity.
  - Inception Score (optional).
  - Structural similarity against reference drawings.
- Gather human feedback from IP professionals on clarity and compliance.

## Infrastructure & Hardware

- Target training hardware: >= 24 GB GPU (A10G) or multi-GPU (A100) for faster epochs.
- QLoRA reduces memory by quantizing base model weights to 4-bit.
- Use `accelerate config` to define multi-GPU or multi-node setups.

## Next Steps

1. Populate `configs/dataset.yaml` with schema matching your data.
2. Implement data ingestion in `src/tuning/dataset.py`.
3. Integrate actual claim parsing pipeline (PDF → text) if needed.
4. Acquire GPU resources and test training loop with a small subset.
5. Iterate on evaluation metrics and prompt templates.

