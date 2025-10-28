from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, Features, Value
from omegaconf import DictConfig

from . import config as path_config
from .utils import ensure_dir, load_config

logger = logging.getLogger(__name__)


@dataclass
class DatasetPaths:
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path


class ClaimsDatasetBuilder:
    """Prepare Hugging Face datasets from raw patent claim inputs."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        storage_cfg = config.get("storage")
        if storage_cfg is None:
            self.paths = DatasetPaths(
                raw_dir=path_config.CLAIMS_RAW_DIR,
                processed_dir=path_config.CLAIMS_PROCESSED_DIR,
                cache_dir=path_config.CLAIMS_CACHE_DIR,
            )
        else:
            self.paths = DatasetPaths(
                raw_dir=Path(storage_cfg.get("raw_dir", path_config.CLAIMS_RAW_DIR)),
                processed_dir=Path(storage_cfg.get("processed_dir", path_config.CLAIMS_PROCESSED_DIR)),
                cache_dir=Path(storage_cfg.get("cache_dir", path_config.CLAIMS_CACHE_DIR)),
            )
        ensure_dir(self.paths.processed_dir)
        ensure_dir(self.paths.cache_dir)

    @classmethod
    def from_config(cls, config_path: Path) -> "ClaimsDatasetBuilder":
        cfg = load_config(config_path)
        return cls(cfg)

    def build(self) -> DatasetDict:
        """Entry point to materialize the dataset."""
        dataset_dir = self.paths.processed_dir / "hf_dataset"
        if dataset_dir.exists():
            logger.info("Loading cached dataset from %s", dataset_dir)
            return DatasetDict.load_from_disk(str(dataset_dir))
        features = Features({
            "id": Value("string"),
            "prompt": Value("string"),
            "negative_prompt": Value("string"),
            "reference_path": Value("string"),
        })
        records = list(self._load_raw_records(self.paths.raw_dir.glob("*.jsonl")))
        dataset = Dataset.from_list(records, features=features)

        split_config = self.config.get("splits", {"train": 0.8, "validation": 0.1, "test": 0.1})
        dataset = dataset.shuffle(seed=int(self.config.get("seed", 42)))
        split_dataset = dataset.train_test_split(test_size=split_config.get("test", 0.1))

        val_fraction = split_config.get("validation", 0.1)
        validation_split = split_dataset["train"].train_test_split(test_size=val_fraction)
        result = DatasetDict({
            "train": validation_split["train"],
            "validation": validation_split["test"],
            "test": split_dataset["test"],
        })

        logger.info("Persisting dataset to %s", dataset_dir)
        ensure_dir(dataset_dir)
        result.save_to_disk(str(dataset_dir))
        return result

    def _load_raw_records(self, files: Iterable[Path]) -> Iterable[dict]:
        for path in files:
            logger.info("Reading raw claims from %s", path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    yield self._build_prompt(record)

    def _build_prompt(self, record: dict) -> dict:
        prompt_template = self.config.get("prompt_template", {}).get("base", "{claim_text}")
        negative_prompt = self.config.get("prompt_template", {}).get("negative", "")
        main_feature = record.get("main_feature") or record.get("category") or "subject"
        prompt = prompt_template.format(
            category=record.get("category", "device"),
            main_feature=main_feature,
            claim_text=record.get("claim_text", ""),
        )
        return {
            "id": record.get("id", ""),
            "prompt": prompt,
            "negative_prompt": record.get("negative_prompt", negative_prompt),
            "reference_path": record.get("reference_path", ""),
        }


def load_dataset_dict(config_path: Path) -> DatasetDict:
    """Convenience helper for CLI usage."""
    builder = ClaimsDatasetBuilder.from_config(config_path)
    return builder.build()
