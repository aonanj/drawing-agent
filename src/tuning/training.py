from __future__ import annotations

import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import DatasetDict
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model

from .dataset import load_dataset_dict
from .utils import ensure_dir, flatten_config, load_config

logger = logging.getLogger(__name__)


class SDXLQLoraTrainer:
    """Bootstrap SDXL pipeline with QLoRA adapters and prepare for fine-tuning."""

    def __init__(self, config: DictConfig, dataset: DatasetDict) -> None:
        self.config = config
        self.dataset = dataset
        training_cfg = config.get("training", {})
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            mixed_precision=training_cfg.get("mixed_precision", "bf16"),
            log_with=config.get("logging", {}).get("report_to"),
            project_dir=config.get("output_dir", "outputs"),
        )
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(config.get("project_name", "drawing-agent"),
                                           config=flatten_config(config))
        self.pipeline = None
        self.unet = None
        self.text_encoders = []

    def setup_models(self) -> None:
        model_cfg = self.config.get("model", {})
        logger.info("Loading SDXL pipeline from %s", model_cfg.get("pretrained_model_name_or_path"))
        torch_dtype = getattr(torch, self.config.get("qlora", {}).get("compute_dtype", "bf16"))
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_cfg.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"),
            torch_dtype=torch_dtype,
            variant=model_cfg.get("variant", "fp16"),
            use_safetensors=True,
        )
        self.unet = self.pipeline.unet
        self.text_encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2]

        qlora_cfg = self.config.get("qlora", {})
        lora_config = LoraConfig(
            r=qlora_cfg.get("r", 32),
            lora_alpha=qlora_cfg.get("alpha", 16),
            target_modules=qlora_cfg.get("target_modules", ["to_k", "to_q"]),
            lora_dropout=qlora_cfg.get("dropout", 0.05),
            bias="none",
        )
        logger.info("Applying QLoRA adapters with rank %s", lora_config.r)
        self.unet = get_peft_model(self.unet, lora_config)
        self.pipeline.unet = self.unet

        if qlora_cfg.get("use_gradient_checkpointing", True):
            enable_gradient_checkpointing = getattr(self.unet, "enable_gradient_checkpointing", None)
            if callable(enable_gradient_checkpointing):
                enable_gradient_checkpointing()
            for encoder in self.text_encoders:
                if hasattr(encoder, "gradient_checkpointing_enable"):
                    encoder.gradient_checkpointing_enable()

        self.pipeline.to(self.accelerator.device)
        ensure_dir(Path(self.config.get("output_dir", "outputs")))

    def train(self) -> None:
        logger.warning("Training loop is a stub. Implement fine-tuning logic before running.")

    def finalize(self) -> None:
        if self.accelerator.is_main_process:
            self.accelerator.end_training()


def train(config_path: Path) -> None:
    config = load_config(config_path)
    dataset_cfg_path = Path(config.get("data", {}).get("dataset_config", "configs/dataset.yaml"))
    dataset = load_dataset_dict(dataset_cfg_path)

    trainer = SDXLQLoraTrainer(config, dataset)
    trainer.setup_models()
    trainer.train()
    trainer.finalize()
