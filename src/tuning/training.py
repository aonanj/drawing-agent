from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import concatenate_datasets, DatasetDict, Dataset as HFDataset
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.optimization import get_scheduler
import gc
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from . import config as path_config
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
            project_dir=config.get("output_dir", str(path_config.OUTPUTS_DIR)),
        )
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(config.get("project_name", "drawing-agent"),
                                           config=flatten_config(config))
        self.pipeline = None
        self.unet = None
        self.text_encoders = []
        self.tokenizers = []
        self.vae = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

    def setup_models(self) -> None:
        model_cfg = self.config.get("model", {})
        logger.info("Loading SDXL pipeline from %s", model_cfg.get("pretrained_model_name_or_path"))

        # Map config dtype strings to torch dtype attributes
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype_str = self.config.get("qlora", {}).get("compute_dtype", "bf16")
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_cfg.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"),
            torch_dtype=torch_dtype,
            variant=model_cfg.get("variant", "fp16"),
            use_safetensors=True,
            cache_dir=model_cfg.get("cache_dir"),  # <--- THIS IS THE FIX
        )
        self.unet = self.pipeline.unet
        self.text_encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
        self.tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2]
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.pipeline.scheduler

        # Freeze VAE and text encoders
        self.vae.requires_grad_(False)
        for encoder in self.text_encoders:
            encoder.requires_grad_(False)

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
        ensure_dir(Path(self.config.get("output_dir", str(path_config.OUTPUTS_DIR))))

    def _preprocess_images(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess images and prompts for training."""
        data_cfg = self.config.get("data", {})
        training_cfg = self.config.get("training", {})
        image_column = data_cfg.get("image_column", "image")
        prompt_column = data_cfg.get("prompt_column", "prompt")
        resolution = training_cfg.get("resolution", 1024)

        # Image preprocessing
        image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        images = [image_transforms(img.convert("RGB")) for img in examples[image_column]]
        prompts = examples[prompt_column]

        return {"pixel_values": images, "prompts": prompts}

    def _encode_prompt(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts using both SDXL text encoders."""
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.accelerator.device)

            with torch.no_grad():
                prompt_embeds = text_encoder(
                    text_input_ids,
                    output_hidden_states=True,
                )

            # Use pooled output from second text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        # Concatenate embeddings from both text encoders
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds_list[1]  # Use pooled from text_encoder_2

        return prompt_embeds, pooled_prompt_embeds

    def _setup_optimizer(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        if self.unet is None:
            raise RuntimeError("setup_models() must be called before _setup_optimizer()")
        
        optimizer_cfg = self.config.get("optimizer", {})
        training_cfg = self.config.get("training", {})

        # Get trainable parameters (only LoRA adapters)
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

        optimizer_type = optimizer_cfg.get("type", "adamw_8bit").lower()
        learning_rate = float(optimizer_cfg.get("learning_rate", 1e-4))

        if "8bit" in optimizer_type:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize,
                    lr=learning_rate,
                    betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
                    weight_decay=optimizer_cfg.get("weight_decay", 0.01),
                )
            except ImportError:
                logger.warning("bitsandbytes not found, falling back to regular AdamW")
                self.optimizer = torch.optim.AdamW(
                    params_to_optimize,
                    lr=learning_rate,
                    betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
                    weight_decay=optimizer_cfg.get("weight_decay", 0.01),
                )
        else:
            self.optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=learning_rate,
                betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
                weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            )

        # Set up learning rate scheduler
        max_train_steps = training_cfg.get("max_train_steps", 1000)
        warmup_steps = optimizer_cfg.get("warmup_steps", 100)
        lr_scheduler_type = optimizer_cfg.get("lr_scheduler", "cosine_with_restarts")
        num_cycles = optimizer_cfg.get("num_cycles", 1)

        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=num_cycles,
        )

    def train(self) -> None:
        """Main training loop for SDXL QLoRA fine-tuning."""
        if self.vae is None or self.unet is None or self.noise_scheduler is None:
            raise RuntimeError("setup_models() must be called before train()")
        
        training_cfg = self.config.get("training", {})
        data_cfg = self.config.get("data", {})
        logging_cfg = self.config.get("logging", {})

        # Configuration
        train_batch_size = training_cfg.get("train_batch_size", 1)
        gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 8)
        max_train_steps = training_cfg.get("max_train_steps", 1000)
        checkpointing_steps = training_cfg.get("checkpointing_steps", 100)
        logging_steps = logging_cfg.get("logging_steps", 10)
        save_adapters_only = training_cfg.get("save_adapters_only", True)



        # Create data loader with custom collation
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            prompts = [example["prompts"] for example in examples]
            return {"pixel_values": pixel_values, "prompts": prompts}

        # Prepare dataset
        base_train_dataset = self.dataset[data_cfg.get("train_split", "train")]
        
        # --- SHARDING & GARBAGE COLLECTION FIX ---
        num_shards = 10  # Increased to 10 for smaller, safer chunks
        processed_shard_paths = []
        
        model_cache_dir = Path(self.config.get("model", {}).get("cache_dir", ".cache/huggingface"))
        dataset_map_cache_dir = model_cache_dir / "datasets_cache"
        ensure_dir(dataset_map_cache_dir)
        
        logger.info(f"Mapping dataset in {num_shards} shards to force garbage collection...")
        
        for i in range(num_shards):
            logger.info(f"--- Processing Shard {i+1} of {num_shards} ---")
            
            # Get one shard
            shard = base_train_dataset.shard(num_shards=num_shards, index=i, contiguous=True)
            
            # Define a unique cache file FOR THIS SHARD on Google Drive
            shard_cache_file = dataset_map_cache_dir / f"train_map_shard_{i}_of_{num_shards}.arrow"
            processed_shard_paths.append(str(shard_cache_file))
            
            # Map just this one shard, writing to its own cache file
            processed_shard = shard.map(
                self._preprocess_images,
                batched=True,
                batch_size=train_batch_size,
                remove_columns=base_train_dataset.column_names,
                cache_file_name=str(shard_cache_file),
            )
            
            # --- This is the new, critical part ---
            logger.info(f"Shard {i+1} mapped. Deleting objects to free VM disk space.")
            del shard
            del processed_shard
            gc.collect()
            logger.info(f"--- Shard {i+1} complete and memory freed. ---")

        logger.info("All shards mapped. Reloading from cache files on Google Drive...")
        
        # Reload all the processed shards from their cache files
        processed_shards = [
            HFDataset.from_file(path) for path in processed_shard_paths
        ]
        
        train_dataset = concatenate_datasets(processed_shards)
        logger.info("All shards concatenated into final training set.")
        # --- END SHARDING & GARBAGE COLLECTION FIX ---

        # Create data loader with custom collation

        # Create data loader with custom collation

        train_dataset.set_format(type="torch")

        # Wrap in a PyTorch Dataset for type compatibility
        class HFDatasetWrapper(Dataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset
            
            def __len__(self):
                return len(self.hf_dataset)
            
            def __getitem__(self, idx):
                return self.hf_dataset[idx]

        wrapped_dataset = HFDatasetWrapper(train_dataset)

        train_dataloader = DataLoader(
            wrapped_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 for debugging, increase for performance
        )

        # Set up optimizer
        self._setup_optimizer()

        # Prepare for distributed training
        self.unet, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, self.lr_scheduler
        )

        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Batch size per device = {train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

        # Training loop
        progress_bar = tqdm(
            range(max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )

        for epoch in range(num_train_epochs):
            self.unet.train()

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Encode images to latent space
                    pixel_values = batch["pixel_values"].to(
                        self.accelerator.device, dtype=self.vae.dtype
                    )

                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample random timesteps
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Encode prompts
                    prompt_embeds, pooled_prompt_embeds = self._encode_prompt(batch["prompts"])

                    # Prepare added time embeddings for SDXL
                    add_time_ids = self._get_add_time_ids(
                        (training_cfg.get("resolution", 1024), training_cfg.get("resolution", 1024)),
                        (0, 0),
                        (training_cfg.get("resolution", 1024), training_cfg.get("resolution", 1024)),
                        bsz,
                    )

                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={
                            "time_ids": add_time_ids,
                            "text_embeds": pooled_prompt_embeds,
                        },
                    ).sample

                    # Calculate loss
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    # MSE loss with optional SNR weighting
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Backpropagate
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # Logging
                    if self.global_step % logging_steps == 0:
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }
                        progress_bar.set_postfix(logs)
                        self.accelerator.log(logs, step=self.global_step)

                    # Checkpointing
                    if self.global_step % checkpointing_steps == 0:
                        self._save_checkpoint(save_adapters_only)

                    # Stop if we've reached max steps
                    if self.global_step >= max_train_steps:
                        break

            if self.global_step >= max_train_steps:
                break

        # Save final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_checkpoint(save_adapters_only, is_final=True)
            logger.info("Training completed!")

    def _get_add_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int],
        target_size: tuple[int, int],
        batch_size: int,
    ) -> torch.Tensor:
        """Prepare time IDs for SDXL conditioning."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids] * batch_size, device=self.accelerator.device)
        return add_time_ids

    def _save_checkpoint(self, save_adapters_only: bool = True, is_final: bool = False) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.config.get("output_dir", str(path_config.OUTPUTS_DIR)))

        if is_final:
            save_path = output_dir / "final"
        else:
            save_path = output_dir / f"checkpoint-{self.global_step}"

        ensure_dir(save_path)

        if save_adapters_only:
            # Save only LoRA adapters
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(save_path / "unet_lora")
            logger.info(f"Saved LoRA adapters to {save_path}")
        else:
            # Save full pipeline
            if self.pipeline is None:
                raise RuntimeError("Pipeline is not initialized. Call setup_models() first.")
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            self.pipeline.unet = unwrapped_unet
            self.pipeline.save_pretrained(save_path)
            logger.info(f"Saved full pipeline to {save_path}")

    def finalize(self) -> None:
        if self.accelerator.is_main_process:
            self.accelerator.end_training()


def train(config_path: Path) -> None:
    config = load_config(config_path)
    dataset_cfg_path = Path(config.get("data", {}).get("dataset_config", path_config.CONFIG_DIR / "dataset.yaml"))
    dataset_config = load_config(dataset_cfg_path)

    # Check if this is a pre-processed dataset (like figures) or needs building (like claims)
    processed_dir = Path(dataset_config.get("storage", {}).get("processed_dir", ""))
    dataset_dir = processed_dir if processed_dir.exists() else processed_dir / "hf_dataset"

    if dataset_dir.exists():
        logger.info("Loading pre-processed dataset from %s", dataset_dir)
        dataset = DatasetDict.load_from_disk(str(dataset_dir))
    else:
        logger.info("Building dataset from raw data using config %s", dataset_cfg_path)
        dataset = load_dataset_dict(dataset_cfg_path)

    trainer = SDXLQLoraTrainer(config, dataset)
    trainer.setup_models()
    trainer.train()
    trainer.finalize()