# drawing-agent: SDXL QLoRA Fine-Tuning for Patent Drawings

Generate patent-style technical drawings from text using fine-tuned Stable Diffusion XL with QLoRA. This project provides a complete pipeline for processing USPTO patent data and training a model to generate patent-compliant line art.

## ➣ Features

- **Automated USPTO Data Processing**: Parse XML patent files, extract claims and descriptions, process multi-figure TIFF images
- **Intelligent Image Processing**: OCR-based figure detection, binarization, deskewing, denoising, and splitting of multi-figure sheets
- **Control Map Generation**: Canny edge detection for ControlNet guidance
- **Structured Prompt Generation**: Diagram type detection, spatial relations extraction, and multi-level context assembly
- **SQLite-based Indexing**: Efficient tracking of processed patents with full-text search capability
- **Family-based Splitting**: Prevent data leakage with patent family-aware train/val/test splits
- **QLoRA Fine-tuning**: Memory-efficient 4-bit quantized training with LoRA adapters
- **Make-based Workflow**: Modular pipeline with incremental processing and database management

## ➣ Quick Start

```bash
# 1. Setup directories and database
make dirs
make init-db

# 2. Add USPTO patent data to data/raw/
# Download from: https://bulkdata.uspto.gov/
# Place .zip files containing individual patents in data/raw/

# 3. Index patent files into database
make index

# 4. Build training datasets
make dataset

# 5. Configure accelerate for training
accelerate config
# Save to: configs/accelerate/default_config.yaml

# 6. Train the model
bash scripts/run_finetune.sh
```

## ➣ Repository Layout

```
drawing-agent/
├── configs/                      # Configuration files
│   ├── dataset.figures.yaml     # Figures dataset config
│   ├── dataset.yaml             # Claims dataset config
│   ├── training.sdxl-qlora.yaml # Training hyperparameters
│   └── accelerate/              # Accelerate configs
├── data/
│   ├── raw/                     # USPTO bulk downloads (zipped patents)
│   ├── work/                    # Working directory
│   │   ├── index.sqlite         # Patent index database
│   │   ├── extracted/           # Extracted XML and TIFF files
│   │   ├── images/              # Processed figure crops
│   │   └── control/             # Canny edge control maps
│   └── ds/                      # Output JSONL datasets
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── scripts/
│   ├── setup_environment.sh     # Environment setup
│   ├── process_uspto.sh         # Process USPTO data
│   ├── run_finetune.sh          # Launch training
│   └── test_setup.sh            # Validate setup
├── src/training/                # Core Python modules
│   ├── index_docs.py            # ZIP indexing and extraction
│   ├── parse_xml.py             # USPTO XML parsing
│   ├── img_norm.py              # Image processing & OCR-based figure detection
│   ├── control.py               # Canny edge control map generation
│   ├── prompt.py                # Structured prompt building
│   ├── build_dataset.py         # Dataset construction from indexed patents
│   ├── load_fts.py              # Full-text search index loader
│   └── schema.sql               # SQLite database schema
├── outputs/                     # Model checkpoints and LoRA adapters
├── Makefile                     # Build automation
├── pyproject.toml               # Python package configuration
└── README.md                    # This file
```

## ➣ Makefile Commands

```bash
# Database setup
make dirs            # Create required directories
make init-db         # Initialize SQLite schema
make index           # Index ZIP bundles into database
make reindex         # Re-extract and re-index with --force

# Dataset building
make train           # Build train.jsonl
make val             # Build val.jsonl
make test            # Build test.jsonl
make dataset         # Build all splits

# Optional utilities
make fts             # Load full-text search index
make check           # Show database statistics
make counts          # Count rows per table
make vacuum          # Optimize database
make clean-outputs   # Remove generated images and JSONL files
make clean-all       # Remove database, extracted files, and outputs

```

## ➣ Dataset Structure

Each training sample in the JSONL output includes:

```json
{
  "id": "US1234567:page123_FIG_3A.png",
  "image_path": "data/work/images/US1234567_page123_FIG_3A.png",
  "control_path": "data/work/control/US1234567_page123_FIG_3A_canny.png",
  "prompt": "Style: USPTO patent line art...",
  "doc_id": "US1234567",
  "fig_label": "FIG. 3A",
  "bbox": [x1, y1, x2, y2],
  "original_size": [width, height],
  "resize_meta": {
    "orig_width": 1500,
    "orig_height": 1200,
    "resized_width": 2048,
    "resized_height": 1638,
    "offset_x": 0,
    "offset_y": 205,
    "scale": 1.365,
    "canvas_size": [2048, 2048]
  }
}
```

**Key Fields:**
- **image_path**: Processed patent figure (2048x2048, monochrome, letterboxed with white padding)
- **control_path**: Canny edge map for ControlNet guidance
- **prompt**: Structured prompt with diagram type, spatial relations, and style constraints
- **resize_meta**: Metadata for reversing the letterbox transformation
- **bbox**: Figure bounding box coordinates from OCR-detected labels

## ➣ Structured Prompts

Prompts are hierarchically constructed by `prompt.py`:

```
Style: USPTO patent line art, monochrome, 300dpi, white background.
Type: block_diagram

Primary Context:
FIG. 3A illustrates a system architecture showing processor-memory connections.

Secondary Context:  
The system comprises a processor 100, memory 110, and bus 120 for data transfer.

Spatial Relations:
- processor 100 connected to memory 110
- bus 120 between processor 100 and memory 110

Prohibitions: no shading, no color, no text outside reference numerals
```

**Diagram Type Detection:**
Automatically classified based on text analysis: flowchart, block diagram, mechanical, 
electrical schematic, graph, perspective, orthographic

## ➣ Configuration

### Dataset (`configs/dataset.figures.yaml`)

```yaml
storage:
  processed_dir: data/figures/processed/hf_dataset

schema:
  prompt: string
  figure_no: string
  diagram_type: string
  patent_id: string
  image: Image
  control: Image

splits:
  train: train
  validation: validation
  test: test

image_processing:
  resolution: 2048
  center_crop: true
  random_flip: false
```

### Training (`configs/training.sdxl-qlora.yaml`)

```yaml
model:
  pretrained_model_name_or_path: stabilityai/stable-diffusion-xl-base-1.0
  revision: fp16
  variant: fp16

qlora:
  r: 32                          # LoRA rank
  alpha: 16
  dropout: 0.05
  target_modules: [to_k, to_q, to_v, to_out.0]
  quantization_bits: 4
  compute_dtype: bf16

training:
  resolution: 2048
  train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  checkpointing_steps: 500
  learning_rate: 1.0e-4
  lr_scheduler: cosine_with_restarts
```

## ➣ Pipeline Overview

### 1. Indexing (`index_docs.py`)
- Scans `data/raw/` for `.zip` files containing patents
- Validates: exactly 1 XML + at least 1 TIFF per zip
- Extracts files to `data/work/extracted/`
- Records paths in SQLite database (`data/work/index.sqlite`)

### 2. XML Parsing (`parse_xml.py`)
- Extracts figure descriptions, claims, and titles from patent XML
- Supports multiple USPTO Red Book formats
- Identifies method claims for flowchart detection
- Returns structured metadata for prompt generation

### 3. Image Processing (`img_norm.py`)
- **OCR-based Figure Detection**: Uses Tesseract to locate "FIG." labels on TIFF pages
- **Figure Extraction**: Analyzes horizontal gaps to split multi-figure sheets
- **Binarization**: Otsu thresholding for clean monochrome line art
- **Deskewing**: Auto-correction using OpenCV minAreaRect
- **Denoising**: Morphological operations + connected components filtering
- **Letterboxing**: Pads to 2048x2048 with white background, preserving metadata for reversal

### 4. Control Maps (`control.py`)
- Generates Canny edge maps (60/180 thresholds)
- Perfect for ControlNet conditioning during training

### 5. Prompt Generation (`prompt.py`)
- **Diagram Type Detection**: Analyzes text for flowchart, block diagram, mechanical, etc.
- **Spatial Relations**: Extracts "connected to", "between", "adjacent to" relationships
- **Hierarchical Assembly**: Combines figure descriptions, claims, and spatial context
- **Style Enforcement**: Adds USPTO-specific constraints and prohibitions

### 6. Dataset Building (`build_dataset.py`)
- Combines processed images, control maps, and prompts
- Generates JSONL files for train/val/test splits
- Includes bounding boxes and resize metadata for each figure

## ➣ Hardware Requirements

**Minimum:**
- 1× GPU with 24GB VRAM (RTX 4090, A10G)
- 32GB RAM
- 100GB SSD storage (for processing)

**Recommended:**
- 1× A100 (40GB) or H100
- 64GB RAM
- 500GB NVMe SSD

**Processing Performance:**
- ~2-5 seconds per patent
- 1000 patents: ~30-90 minutes
- Training: ~2-8 hours for 3 epochs

## ➣ Key Technologies

### Core Dependencies
- **PyTorch 2.9+**: Deep learning framework
- **Diffusers 0.35+**: SDXL model and training utilities
- **PEFT 0.17+**: Parameter-efficient fine-tuning (LoRA/QLoRA)
- **Accelerate 1.11+**: Distributed training and mixed precision
- **BitsAndBytes 0.42+**: 4-bit quantization for QLoRA

### Image Processing
- **OpenCV**: Image manipulation, binarization, deskewing
- **Pytesseract**: OCR for figure label detection
- **Pillow**: Image I/O and basic processing
- **scikit-image**: Advanced image processing utilities

### Data Management
- **SQLite3**: Patent index and full-text search
- **lxml**: Fast XML parsing
- **Datasets**: HuggingFace dataset loading and processing

## ➣ Model Inference

After training, generate patent-style drawings using the fine-tuned model:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load base model with LoRA adapters
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipeline.load_lora_weights("outputs/sdxl-qlora-run-3/checkpoint-3500/unet_lora")
pipeline.to("cuda")

# Generate patent drawing
prompt = """
Style: USPTO patent line art, monochrome, 300dpi, white background.
Type: block_diagram

Primary Context:
System architecture showing processor-memory connections.

Spatial Relations:
- processor 100 connected to memory 110
- bus 120 between processor 100 and memory 110

Prohibitions: no shading, no color, no text outside reference numerals
"""

image = pipeline(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
image.save("patent_drawing.png")
```

## ➣ Database Schema

The SQLite database (`data/work/index.sqlite`) tracks all processed patents:

```sql
-- Core document index
CREATE TABLE docs (
  doc_id TEXT PRIMARY KEY,    -- e.g., US20230001234A1
  xml    TEXT NOT NULL,       -- path to extracted XML
  tiffs  TEXT NOT NULL        -- JSON array of TIFF paths
);

-- Optional figure registry
CREATE TABLE figures (
  id        TEXT PRIMARY KEY, -- {doc_id}:{page}:{label}
  doc_id    TEXT NOT NULL,
  tiff_path TEXT NOT NULL,
  fig_label TEXT,             -- e.g., "FIG. 3A"
  bbox_x1   INTEGER,
  bbox_y1   INTEGER,
  bbox_x2   INTEGER,
  bbox_y2   INTEGER
);

-- Full-text search (optional)
CREATE VIRTUAL TABLE text_fts USING fts5(
  doc_id UNINDEXED,
  section,    -- 'caption' | 'paragraph' | 'claim'
  content
);
```

## ➣ Troubleshooting

**Index fails with "multiple XML files" error:**
- Each ZIP should contain exactly 1 XML file
- Check for malformed or duplicate XMLs in the patent package

**OCR doesn't detect figures:**
- Verify TIFF resolution is 300+ DPI
- Check that "FIG." labels are clearly visible in the image
- Try adjusting Tesseract PSM mode in `img_norm.py`

**CUDA out of memory during training:**
- Reduce `train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 16
- Enable `use_cpu_offload: true` in training config
- Use `enable_xformers_memory_efficient_attention: true`

**No figures split from multi-figure sheets:**
- Adjust `min_gap` parameter in figure splitting logic
- Check TIFF has sufficient white space between figures
- Verify binarization threshold in `img_norm.py`

**Database locked errors:**
- Close any open connections to `index.sqlite`
- Run `make vacuum` to optimize database
- Check WAL files aren't corrupted (`.sqlite-shm`, `.sqlite-wal`)

## ➣ Dataset Inspection

Query the database to inspect indexed patents:

```bash
# Count documents
sqlite3 data/work/index.sqlite "SELECT COUNT(*) FROM docs;"

# List first 10 patents
sqlite3 data/work/index.sqlite "SELECT doc_id FROM docs LIMIT 10;"

# Search for specific patent
sqlite3 data/work/index.sqlite "SELECT * FROM docs WHERE doc_id='US12345678';"

# Check figure count (if figures table populated)
sqlite3 data/work/index.sqlite "SELECT COUNT(*) FROM figures;"
```

View generated JSONL samples:

```bash
# Inspect first training sample
head -n 1 data/ds/train.jsonl | jq '.'

# Count samples per split
wc -l data/ds/*.jsonl
```

## ➣ Project Status

This project is in active development. Current checkpoints available:
- `outputs/sdxl-qlora-run-3/checkpoint-3500/`: Latest trained model
- Multiple intermediate checkpoints at 500-step intervals

## ➣ License

This repository is publicly viewable for portfolio purposes only. The code is proprietary.
Copyright © 2025 Phaethon Order LLC. All rights reserved.
See [LICENSE](LICENSE.md) for terms.

## ➣ Resources

- **USPTO Bulk Data**: https://bulkdata.uspto.gov/
- **SDXL Base Model**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Accelerate Guide**: https://huggingface.co/docs/accelerate
- **Diffusers Training**: https://huggingface.co/docs/diffusers/training/overview

