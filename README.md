# drawing-agent: SDXL QLoRA Fine-Tuning for Patent Drawings

Generate patent-style technical drawings from text using fine-tuned Stable Diffusion XL with QLoRA. This project provides a complete pipeline for processing USPTO patent data and training a model to generate patent-compliant line art.

## ➣ Features

- **Automated USPTO Data Processing**: Parse XML patent files, extract claims and descriptions, process multi-figure TIFF images
- **Intelligent Image Processing**: Binarization, deskewing, denoising, and splitting of multi-figure sheets
- **Control Map Generation**: Canny edge detection for ControlNet guidance
- **Structured Prompt Generation**: Combine figure descriptions, claims, and technical constraints
- **Family-based Splitting**: Prevent data leakage with patent family-aware train/val/test splits
- **QLoRA Fine-tuning**: Memory-efficient 4-bit quantized training with LoRA adapters
- **CLI Tools**: Easy-to-use commands for processing and training

## ➣ Quick Start

```bash
# 1. Setup environment
bash scripts/setup_environment.sh
source .venv/drawing-agent/bin/activate

# 2. Verify setup
bash scripts/test_setup.sh

# 3. Add USPTO patent data to data/raw/
# Download from: https://bulkdata.uspto.gov/

# 4. Process patents into training dataset
bash scripts/process_uspto.sh

# 5. Configure accelerate
accelerate config
# Save to: configs/accelerate/default_config.yaml

# 6. Train the model
bash scripts/run_finetune.sh
```

📖 **See [QUICKSTART.md](QUICKSTART.md) for detailed instructions**

## ➣ Test Single Patent

Before processing large batches, test with a single patent:

```bash
# Test a single patent zip
python test_single_patent.py data/raw/redbook_2024_01/US12345678.zip

# Outputs saved to test_output/
# - Processed images
# - Control maps
# - Generated prompts
# - Metadata JSON
```

## ➣ Repository Layout

```
drawing-agent/
├── configs/                      # Configuration files
│   ├── figures.yaml             # USPTO processing config
│   ├── dataset.yaml             # Claims dataset config
│   ├── training.sdxl-qlora.yaml # Training hyperparameters
│   └── accelerate/              # Accelerate configs
├── data/
│   ├── raw/                     # USPTO bulk downloads (zipped patents)
│   ├── figures/                 # Processed patent figures
│   │   ├── processed/           # HuggingFace dataset
│   │   │   ├── hf_dataset/     # Train/val/test splits
│   │   │   ├── images/         # Processed images + control maps
│   │   │   └── dataset_summary.json
│   │   └── cache/              # Temporary processing cache
│   └── claims/                  # Alternative claims-based dataset
├── docs/
│   ├── blueprint.md             # Architecture overview
│   ├── USPTO_PROCESSING_GUIDE.md # Detailed processing guide
│   └── research-notes.md
├── scripts/
│   ├── setup_environment.sh     # Environment setup
│   ├── process_uspto.sh         # Process USPTO data
│   ├── run_finetune.sh          # Launch training
│   └── test_setup.sh            # Validate setup
├── src/tuning/                  # Python package
│   ├── cli.py                   # CLI entrypoints
│   ├── xml_parser.py            # USPTO XML parsing
│   ├── image_processing.py      # TIFF processing & control maps
│   ├── patent_figures.py        # Main processing orchestration
│   ├── dataset.py               # Dataset builders
│   ├── training.py              # QLoRA training loop
│   └── utils.py                 # Helper functions
├── test_single_patent.py        # Single patent test script
├── QUICKSTART.md                # Step-by-step guide
└── README.md                    # This file
```

## ➣ CLI Commands

```bash
# Process USPTO patent figures
python -m src.tuning.cli process-figures --config configs/figures.yaml

# Prepare claims dataset (alternative)
python -m src.tuning.cli prepare-claims --config configs/dataset.yaml

# Train model
python -m src.tuning.cli train-model --config configs/training.sdxl-qlora.yaml

# Show project status
python -m src.tuning.cli info
```

## ➣ Dataset Structure

Each training sample includes:
- **Target Image**: Processed patent figure (2048x2048, monochrome)
- **Control Map**: Canny edge detection for ControlNet
- **Structured Prompt**:
  ```
  Style: USPTO patent line art, monochrome, 300dpi, white background.
  Figure: FIG3, Type: block_diagram
  Description: System architecture showing processor-memory connections
  Objects: processor 100, memory 110, bus 120
  Relations: processor 100 connected to memory 110 via bus 120
  Prohibitions: no shading, no color, no text outside labels
  ```
- **Metadata**: Patent ID, CPC codes, family ID, claims, abstract
- **Resize Metadata**: Original crop size, expanded bounding box, and scale/offset details for reversing the 2048x2048 letterbox normalization

## ➣ Configuration

### USPTO Processing (`configs/figures.yaml`)

```yaml
storage:
  raw_dir: data/raw              # USPTO bulk downloads
  processed_dir: data/figures/processed
  cache_dir: data/figures/cache

image:
  size: 2048                     # Target image dimensions
  min_gap: 40                    # Gap detection threshold
  gap_threshold: 0.005

splits:
  train: 0.8
  validation: 0.1
  test: 0.1
```

### Training (`configs/training.sdxl-qlora.yaml`)

```yaml
model:
  pretrained_model_name_or_path: stabilityai/stable-diffusion-xl-base-1.0

qlora:
  r: 32                          # LoRA rank
  alpha: 16
  dropout: 0.05
  quantization_bits: 4

training:
  train_batch_size: 1
  gradient_accumulation_steps: 8
  max_train_steps: 1000
  learning_rate: 1.0e-4
```

## ➣ Hardware Requirements

**Minimum:**
- 1× GPU with 24GB VRAM (RTX 4090, A10G)
- 32GB RAM
- 500GB SSD storage

**Recommended:**
- 1× A100 (40GB) or 2× A10G
- 64GB RAM
- 1TB NVMe SSD

**Processing Performance:**
- ~2-5 seconds per patent
- 1000 patents: ~30-90 minutes
- Training: ~2-8 hours for 1000 steps

## ➣ Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Complete workflow from setup to inference
- **[docs/USPTO_PROCESSING_GUIDE.md](docs/USPTO_PROCESSING_GUIDE.md)**: Detailed processing pipeline documentation
- **[docs/blueprint.md](docs/blueprint.md)**: System architecture and design decisions

## ➣ Key Features

### XML Parsing
- Extracts figure descriptions, claims, CPC codes
- Identifies diagram types (block, flowchart, perspective, etc.)
- Parses visualizable claim elements
- Handles multiple USPTO XML formats

### Image Processing
- **Binarization**: Otsu thresholding for clean line art
- **Deskewing**: Auto-correction using minAreaRect
- **Denoising**: Morphological operations + connected components
- **Multi-figure splitting**: Detects horizontal gaps between figures
- **Smart cropping**: Content-aware boundary detection
- **Padding & resizing**: 2048x2048 with white background

### Control Maps
- Canny edge detection (100/200 thresholds)
- Edge dilation for visibility
- Perfect for ControlNet conditioning

### Prompt Engineering
- Hierarchical prompt construction
- Primary: Figure descriptions
- Secondary: Detailed description context  
- Tertiary: Claim-derived constraints
- Style enforcement & prohibitions

## ➣ Quality Validation

Automatic quality checks:
- Resolution validation (min 512×512)
- Monochrome verification
- Content ratio checks (>1% non-white)
- Multi-XML file detection
- Patent family deduplication

## ➣ Model Generation

After training, generate drawings:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load model with LoRA adapters
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipeline.load_lora_weights("outputs/sdxl-qlora/checkpoint-1000")
pipeline.to("cuda")

# Generate
prompt = """
Style: USPTO patent line art, monochrome, 300dpi, white background.
Type: block_diagram
Objects: processor, memory, controller
Relations: processor connected to memory
Prohibitions: no shading, no color
"""

image = pipeline(prompt, num_inference_steps=50).images[0]
image.save("patent_drawing.png")
```

## ➣ Troubleshooting

**CUDA out of memory:**
- Reduce `train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable CPU offload

**No figures found:**
- Check zip contains exactly 1 XML + TIFF files
- Verify TIFF files are valid images
- Run `test_single_patent.py` for debugging

**Poor quality outputs:**
- Review input TIFF resolution (300+ DPI recommended)
- Adjust binarization thresholds
- Check prompt quality in dataset

See [docs/USPTO_PROCESSING_GUIDE.md](docs/USPTO_PROCESSING_GUIDE.md) for more troubleshooting.

## ➣ Dataset Statistics

After processing, view summary:
```bash
cat data/figures/processed/dataset_summary.json
```

Includes:
- Sample counts by split
- Diagram type distribution
- CPC code distribution
- Publication kind stats

## ➣ Contributing

When adding features:
1. Test with single patent first
2. Validate output quality manually
3. Update documentation
4. Run full test suite

## ➣ License

See LICENSE file for details.

## ➣ Resources

- **USPTO Bulk Data**: https://data.uspto.gov/bulkdata/datasets 
- **SDXL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **QLoRA/PEFT**: https://huggingface.co/docs/peft
- **Accelerate**: https://huggingface.co/docs/accelerate

---

**Ready to start?** → [QUICKSTART.md](QUICKSTART.md)
