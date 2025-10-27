from __future__ import annotations

import logging
from pathlib import Path

import typer

from .dataset import ClaimsDatasetBuilder
from .preprocess import clean_claim_text
from .training import train as run_training
from .utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Utilities for SDXL QLoRA patent drawing fine-tuning.")


@app.command()
def preprocess_claim(
    text: str = typer.Argument(..., help="Raw patent claim text to normalize."),
) -> None:
    """Preview the text normalization pipeline from the CLI."""
    typer.echo(clean_claim_text(text))


@app.command()
def build_dataset(
    config: Path = typer.Option(Path("configs/dataset.yaml"), help="Dataset configuration path."),
) -> None:
    """Materialize dataset splits from raw claim JSONL files."""
    builder = ClaimsDatasetBuilder.from_config(config)
    dataset_dict = builder.build()
    for split, ds in dataset_dict.items():
        logger.info("Split %s contains %s records", split, len(ds))


@app.command()
def train(
    config: Path = typer.Option(Path("configs/training.sdxl-qlora.yaml"), help="Training config path."),
) -> None:
    """Launch QLoRA fine-tuning (requires implementation of the training loop)."""
    run_training(config)


@app.command()
def show_config(
    config: Path = typer.Argument(..., help="Config file to display."),
) -> None:
    """Print a YAML config after resolving OmegaConf interpolations."""
    cfg = load_config(config)
    typer.echo(cfg.pretty())


if __name__ == "__main__":
    app()
