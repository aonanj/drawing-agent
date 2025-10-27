from __future__ import annotations

import logging
import inspect
from pathlib import Path
from functools import wraps

import typer
from rich.console import Console
from rich.logging import RichHandler

from .dataset import load_dataset_dict
from .patent_figures import process_uspto_data
from .training import train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="drawing-agent",
    help="SDXL QLoRA fine-tuning CLI for patent drawing generation",
    add_completion=False,
)


@app.command()
def process_figures(
    config: Path = typer.Option(
        Path("configs/figures.yaml"),
        "--config",
        "-c",
        help="Path to figures configuration file",
        exists=True,
    ),
) -> None:
    """
    Process USPTO patent bulk downloads into training dataset.
    
    This command will:
    1. Find all zipped patent files in data/raw/*
    2. Extract and parse XML files
    3. Process TIFF images (split, binarize, deskew, denoise)
    4. Generate control maps (Canny edges)
    5. Create structured prompts from patent text
    6. Save HuggingFace dataset with train/val/test splits
    """
    console.print("[bold blue]Processing USPTO patent figures...[/bold blue]")
    console.print(f"Config: {config}")

    try:
        dataset_dict = process_uspto_data(config)
        
        console.print("\n[bold green]✓ Processing complete![/bold green]")
        console.print(f"Train samples: {len(dataset_dict['train'])}")
        console.print(f"Validation samples: {len(dataset_dict['validation'])}")
        console.print(f"Test samples: {len(dataset_dict['test'])}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Processing failed: {e}[/bold red]")
        logger.exception("Failed to process figures")
        raise typer.Exit(1)


@app.command()
def prepare_claims(
    config: Path = typer.Option(
        Path("configs/dataset.yaml"),
        "--config",
        "-c",
        help="Path to dataset configuration file",
        exists=True,
    ),
) -> None:
    """
    Prepare claims dataset from raw JSONL files.
    
    This is an alternative to process_figures for simpler claim-based datasets.
    """
    console.print("[bold blue]Preparing claims dataset...[/bold blue]")
    console.print(f"Config: {config}")
    
    try:
        dataset_dict = load_dataset_dict(config)
        
        console.print("\n[bold green]✓ Dataset prepared![/bold green]")
        console.print(f"Train samples: {len(dataset_dict['train'])}")
        console.print(f"Validation samples: {len(dataset_dict['validation'])}")
        console.print(f"Test samples: {len(dataset_dict['test'])}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Preparation failed: {e}[/bold red]")
        logger.exception("Failed to prepare claims dataset")
        raise typer.Exit(1)


@app.command()
def train_model(
    config: Path = typer.Option(
        Path("configs/training.sdxl-qlora.yaml"),
        "--config",
        "-c",
        help="Path to training configuration file",
        exists=True,
    ),
) -> None:
    """
    Fine-tune SDXL model with QLoRA adapters.
    
    This command should be run with accelerate:
        accelerate launch -m src.tuning.cli train --config configs/training.sdxl-qlora.yaml
    """
    console.print("[bold blue]Starting SDXL QLoRA training...[/bold blue]")
    console.print(f"Config: {config}")
    
    try:
        train(config)
        console.print("\n[bold green]✓ Training complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Training failed: {e}[/bold red]")
        logger.exception("Training failed")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display project information and status."""
    console.print("[bold]Drawing Agent - Patent Drawing Generation[/bold]\n")
    
    # Check for processed datasets
    figures_dataset = Path("data/figures/processed/hf_dataset")
    claims_dataset = Path("data/claims/processed/hf_dataset")
    
    console.print("[bold]Datasets:[/bold]")
    if figures_dataset.exists():
        console.print(f"  ✓ Figures dataset: {figures_dataset}")
    else:
        console.print("  ✗ Figures dataset not found (run 'process-figures' first)")
    
    if claims_dataset.exists():
        console.print(f"  ✓ Claims dataset: {claims_dataset}")
    else:
        console.print("  ✗ Claims dataset not found")
    
    # Check for raw data
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        bulk_folders = [d for d in raw_dir.iterdir() if d.is_dir()]
        console.print("\n[bold]Raw Data:[/bold]")
        console.print(f"  Found {len(bulk_folders)} bulk download folders in {raw_dir}")
    
    # Check for training outputs
    output_dir = Path("outputs/sdxl-qlora")
    if output_dir.exists():
        checkpoints = list(output_dir.glob("checkpoint-*"))
        console.print("\n[bold]Training Outputs:[/bold]")
        console.print(f"  Found {len(checkpoints)} checkpoints in {output_dir}")


if __name__ == "__main__":
    app()