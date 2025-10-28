from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from omegaconf import DictConfig
from tqdm import tqdm

from . import config as path_config
from .image_processing import TIFFImageProcessor, validate_image_quality
from .utils import ensure_dir, load_config
from .xml_parser import PatentData, USPTOXMLParser, extract_figure_context

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """Single training sample with image, control map, and metadata."""
    image_path: str
    control_path: str
    prompt: str
    figure_no: str
    diagram_type: Optional[str]
    patent_id: str
    pub_kind: str
    family_id: str
    cpc: List[str]
    source: str
    metadata: Dict


class PatentFiguresProcessor:
    """Process USPTO patent bulk downloads into training dataset."""

    def __init__(self, config: DictConfig):
        self.config = config
        storage_cfg = config.get("storage", {})
        
        self.raw_dir = Path(storage_cfg.get("raw_dir", path_config.FIGURES_RAW_DIR))
        self.processed_dir = Path(storage_cfg.get("processed_dir", path_config.FIGURES_PROCESSED_DIR))
        self.cache_dir = Path(storage_cfg.get("cache_dir", path_config.FIGURES_CACHE_DIR))
        
        ensure_dir(self.processed_dir)
        ensure_dir(self.cache_dir)
        
        image_cfg = config.get("image", {})
        self.image_processor = TIFFImageProcessor(
            target_size=image_cfg.get("size", 1024),
            min_gap=image_cfg.get("min_gap", 40),
            gap_threshold=image_cfg.get("gap_threshold", 0.005),
        )
        
        self.seed = config.get("seed", 42)
        self.splits = config.get("splits", {"train": 0.8, "validation": 0.1, "test": 0.1})
        
        # Track families to ensure no leakage between splits
        self.family_samples: Dict[str, List[TrainingSample]] = defaultdict(list)

    @classmethod
    def from_config(cls, config_path: Path) -> "PatentFiguresProcessor":
        """Create processor from config file."""
        cfg = load_config(config_path)
        return cls(cfg)

    def process_bulk_downloads(self) -> DatasetDict:
        """
        Main entry point: process all bulk download folders.
        
        Returns:
            HuggingFace DatasetDict with train/validation/test splits
        """
        # Check for cached dataset
        dataset_path = self.processed_dir / "hf_dataset"
        if dataset_path.exists():
            logger.info("Loading cached dataset from %s", dataset_path)
            try:
                return DatasetDict.load_from_disk(str(dataset_path))
            except Exception as e:
                logger.warning("Failed to load cached dataset, will reprocess: %s", e)
                # Remove corrupted cache
                import shutil
                shutil.rmtree(dataset_path)
        
        # Find all bulk download folders
        bulk_folders = [d for d in self.raw_dir.iterdir() if d.is_dir()]
        logger.info("Found %d bulk download folders", len(bulk_folders))
        
        if not bulk_folders:
            raise ValueError(
                f"No bulk download folders found in {self.raw_dir}. "
                f"Please add USPTO patent data to {path_config.RAW_DATA_DIR}"
            )
        
        # Process each bulk download
        all_samples = []
        for bulk_folder in tqdm(bulk_folders, desc="Processing bulk downloads"):
            samples = self._process_bulk_folder(bulk_folder)
            all_samples.extend(samples)
            logger.info("Processed %s: %d samples", bulk_folder.name, len(samples))
        
        logger.info("Processed %d total samples", len(all_samples))
        
        if not all_samples:
            raise ValueError(
                "No valid samples were generated. Please check:\n"
                "1. Zip files contain exactly 1 XML + TIFF files\n"
                "2. XML files are valid USPTO format\n"
                "3. TIFF images are readable\n"
                "Run test_single_patent.py on a sample file for debugging"
            )
        
        # Organize by family
        for sample in all_samples:
            self.family_samples[sample.family_id].append(sample)
        
        logger.info("Found %d unique patent families", len(self.family_samples))
        
        # Create family-based splits
        dataset_dict = self._create_splits(all_samples)
        
        # Save dataset
        logger.info("Saving dataset to %s", dataset_path)
        # Work around HF datasets bug when saving empty splits by forcing a single shard
        empty_split_shards = {
            str(split): 1 for split, split_dataset in dataset_dict.items() if len(split_dataset) == 0
        }
        if empty_split_shards:
            logger.debug(
                "Empty splits detected (%s); using single shard to avoid save_to_disk division bug",
                ", ".join(str(k) for k in empty_split_shards.keys()),
            )
        dataset_dict.save_to_disk(
            str(dataset_path),
            num_shards=empty_split_shards or None,
        )
        
        # Save metadata summary
        self._save_metadata_summary(all_samples)
        
        return dataset_dict

    def _process_bulk_folder(self, bulk_folder: Path) -> List[TrainingSample]:
        """Process a single bulk download folder containing zipped patents."""
        samples = []
        
        # Find all zip files
        zip_files = list(bulk_folder.glob("*.zip")) + list(bulk_folder.glob("*.ZIP"))
        
        for zip_path in tqdm(zip_files, desc=f"Processing {bulk_folder.name}", leave=False):
            try:
                patent_samples = self._process_patent_zip(zip_path, bulk_folder.name)
                samples.extend(patent_samples)
            except Exception as e:
                logger.error("Failed to process %s: %s", zip_path, e)
        
        return samples

    def _process_patent_zip(self, zip_path: Path, source: str) -> List[TrainingSample]:
        """Process a single patent zip file."""
        samples = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Check for unexpected file extensions
                allowed_extensions = {'.xml', '.tif', '.tiff'}
                all_files = zf.namelist()

                for filename in all_files:
                    # Skip directories
                    if filename.endswith('/'):
                        continue

                    # Get file extension (case-insensitive)
                    ext = Path(filename).suffix.lower()

                    if ext not in allowed_extensions:
                        logger.debug("Skipping %s: contains unexpected file %s (extension: %s)",
                                   zip_path.name, filename, ext)
                        return samples

                # Find XML files
                xml_files = [f for f in all_files if f.endswith(".xml") or f.endswith(".XML")]

                # Skip if multiple XML files (as per requirements)
                if len(xml_files) != 1:
                    if len(xml_files) > 1:
                        logger.debug("Skipping %s: contains %d XML files", zip_path.name, len(xml_files))
                    else:
                        logger.debug("Skipping %s: no XML files found", zip_path.name)
                    return samples
                
                xml_filename = xml_files[0]
                
                # Extract and parse XML
                xml_data = zf.read(xml_filename)
                xml_path = self.cache_dir / f"{zip_path.stem}.xml"
                xml_path.write_bytes(xml_data)
                
                parser = USPTOXMLParser(xml_path)
                patent_data = parser.parse()
                
                if patent_data is None:
                    logger.warning("Failed to parse patent data from %s", zip_path.name)
                    xml_path.unlink(missing_ok=True)
                    return samples
                
                logger.debug("Parsed patent %s: %d figures, %d claims", 
                           patent_data.patent_id, 
                           len(patent_data.figure_descriptions),
                           len(patent_data.claims))
                
                # Find and process TIFF files
                tiff_files = [f for f in zf.namelist() 
                             if f.lower().endswith((".tif", ".tiff"))]
                
                if not tiff_files:
                    logger.debug("No TIFF files found in %s", zip_path.name)
                    xml_path.unlink(missing_ok=True)
                    return samples
                
                logger.debug("Found %d TIFF file(s) in %s", len(tiff_files), zip_path.name)
                
                # Create output directory for this patent
                patent_output_dir = self.processed_dir / "images" / patent_data.patent_id
                ensure_dir(patent_output_dir)
                
                # Process each TIFF
                total_figures = 0
                valid_figures = 0
                for tiff_filename in tiff_files:
                    tiff_data = zf.read(tiff_filename)
                    tiff_path = self.cache_dir / f"{zip_path.stem}_{Path(tiff_filename).name}"
                    tiff_path.write_bytes(tiff_data)
                    
                    # Process TIFF images
                    try:
                        image_pairs = self.image_processor.process_tiff(
                            tiff_path, patent_output_dir, patent_data.patent_id
                        )
                        total_figures += len(image_pairs)
                        
                        # Create training samples
                        for target_path, control_path in image_pairs:
                            # Validate image quality
                            if not validate_image_quality(target_path):
                                logger.debug("Skipping low-quality image %s", target_path.name)
                                continue
                            
                            sample = self._create_training_sample(
                                target_path, control_path, patent_data, source
                            )
                            if sample:
                                samples.append(sample)
                                valid_figures += 1
                    except Exception as e:
                        logger.warning("Failed to process TIFF %s from %s: %s", 
                                     tiff_filename, zip_path.name, e)
                    
                    # Clean up cached TIFF
                    tiff_path.unlink(missing_ok=True)
                
                logger.debug("Patent %s: %d figures extracted, %d valid samples created",
                           patent_data.patent_id, total_figures, valid_figures)
                
                # Clean up cached XML
                xml_path.unlink(missing_ok=True)
                
        except zipfile.BadZipFile:
            logger.error("Bad zip file: %s", zip_path.name)
        except Exception as e:
            logger.error("Error processing zip %s: %s", zip_path.name, e, exc_info=True)
        
        return samples

    def _create_training_sample(
        self,
        target_path: Path,
        control_path: Path,
        patent_data: PatentData,
        source: str,
    ) -> Optional[TrainingSample]:
        """Create a training sample from processed images and patent data."""
        # Try to identify which figure this is
        figure_no = self._identify_figure_number(target_path.stem, patent_data)
        
        # Generate prompt
        prompt = self._generate_prompt(figure_no, patent_data)
        
        if not prompt:
            logger.debug("Could not generate prompt for %s (patent %s)", 
                       target_path.name, patent_data.patent_id)
            return None
        
        # Determine diagram type
        diagram_type = None
        if figure_no and figure_no in patent_data.figure_descriptions:
            diagram_type = patent_data.figure_descriptions[figure_no].diagram_type
        
        # Extract claim elements for metadata
        claim_components = []
        claim_relations = []
        for claim in patent_data.claims[:3]:  # Include top 3 claims
            if claim.is_independent:
                claim_components.extend(claim.components)
                claim_relations.extend(claim.relations)
        
        metadata = {
            "abstract": patent_data.abstract[:500] if patent_data.abstract else "",
            "claim_components": claim_components[:10],
            "claim_relations": claim_relations[:5],
            "figure_description": (
                patent_data.figure_descriptions[figure_no].description
                if figure_no in patent_data.figure_descriptions
                else ""
            ),
        }
        
        sample = TrainingSample(
            image_path=str(target_path.relative_to(self.processed_dir)),
            control_path=str(control_path.relative_to(self.processed_dir)),
            prompt=prompt,
            figure_no=figure_no or "UNKNOWN",
            diagram_type=diagram_type,
            patent_id=patent_data.patent_id,
            pub_kind=patent_data.pub_kind,
            family_id=patent_data.family_id or patent_data.patent_id,
            cpc=patent_data.cpc_codes[:5],  # Limit CPC codes
            source=source,
            metadata=metadata,
        )
        
        logger.debug("Created sample for %s: figure=%s, type=%s", 
                   patent_data.patent_id, figure_no or "UNKNOWN", diagram_type or "unknown")
        
        return sample

    def _identify_figure_number(self, filename: str, patent_data: PatentData) -> Optional[str]:
        """Try to identify figure number from filename or patent data."""
        import re

        # Check if filename contains figure number (e.g., "FIG1", "FIG_1", etc.)
        fig_match = re.search(r"FIG[._\s]*(\d+[A-Z]?)", filename, re.IGNORECASE)
        if fig_match:
            return f"FIG{fig_match.group(1)}"

        # Try to extract index from filename pattern like "patent_p0_f0" or "patent_f0"
        # and map to figure number if we have figure descriptions
        index_match = re.search(r"[_\.]f(\d+)", filename)
        if index_match and patent_data.figure_descriptions:
            fig_index = int(index_match.group(1))
            # Get sorted figure numbers
            sorted_figs = sorted(patent_data.figure_descriptions.keys(),
                               key=lambda x: int(m.group()) if (m := re.search(r'\d+', x)) else 0)
            if fig_index < len(sorted_figs):
                return sorted_figs[fig_index]

        # If just patent_id.png (no index), try to use first figure
        if patent_data.figure_descriptions and not re.search(r"[_\.]f\d+", filename):
            sorted_figs = sorted(patent_data.figure_descriptions.keys(),
                               key=lambda x: int(m.group()) if (m := re.search(r'\d+', x)) else 0)
            return sorted_figs[0] if sorted_figs else None

        # If only one figure description, use that
        if len(patent_data.figure_descriptions) == 1:
            return list(patent_data.figure_descriptions.keys())[0]

        return None

    def _generate_prompt(self, figure_no: Optional[str], patent_data: PatentData) -> str:
        """Generate structured prompt for the figure."""
        prompt_parts = [
            "Style: USPTO patent line art, monochrome, 300dpi, white background."
        ]
        
        # Add figure information
        if figure_no:
            prompt_parts.append(f"Figure: {figure_no}")
            
            # Add figure description if available
            if figure_no in patent_data.figure_descriptions:
                fig_desc = patent_data.figure_descriptions[figure_no]
                if fig_desc.diagram_type:
                    prompt_parts.append(f"Type: {fig_desc.diagram_type}")
                prompt_parts.append(f"Description: {fig_desc.description}")
                
                # Add context from detailed description
                context = extract_figure_context(patent_data, figure_no)
                if context:
                    prompt_parts.append(f"Context: {context}")
        
        # Add claim elements
        if patent_data.claims:
            independent_claims = [c for c in patent_data.claims if c.is_independent]
            if independent_claims:
                claim = independent_claims[0]
                if claim.components:
                    components_str = ", ".join(claim.components[:8])
                    prompt_parts.append(f"Objects: {components_str}")
                if claim.relations:
                    relations_str = "; ".join(claim.relations[:3])
                    prompt_parts.append(f"Relations: {relations_str}")
        
        # Add prohibitions
        prompt_parts.append(
            "Prohibitions: no shading, no color, no text outside labels, "
            "no photo textures, no figure numbers."
        )

        return "\n".join(prompt_parts)

    def _create_splits(self, samples: List[TrainingSample]) -> DatasetDict:
        """Create train/validation/test splits based on patent families."""
        import random
        random.seed(self.seed)
        
        # Group samples by family
        families = list(self.family_samples.keys())
        random.shuffle(families)
        
        # Calculate split sizes
        n_families = len(families)
        n_train = int(n_families * self.splits["train"])
        n_val = int(n_families * self.splits["validation"])
        
        # Split families
        train_families = set(families[:n_train])
        val_families = set(families[n_train:n_train + n_val])
        
        # Organize samples into splits
        train_samples_raw = []
        val_samples_raw = []
        test_samples_raw = []
        
        for sample in samples:
            if sample.family_id in train_families:
                train_samples_raw.append(sample)
            elif sample.family_id in val_families:
                val_samples_raw.append(sample)
            else:
                test_samples_raw.append(sample)
        
        logger.info(
            "Split sizes: %d train, %d val, %d test samples",
            len(train_samples_raw), len(val_samples_raw), len(test_samples_raw)
        )
        
        # Create datasets
        features = Features({
            "image": HFImage(),
            "control": HFImage(),
            "prompt": Value("string"),
            "figure_no": Value("string"),
            "diagram_type": Value("string"),
            "patent_id": Value("string"),
            "pub_kind": Value("string"),
            "family_id": Value("string"),
            "cpc": [Value("string")],
            "source": Value("string"),
            "metadata": Value("string"),  # Store metadata as a JSON string
        })

        def process_and_convert_samples(samples_list: List[TrainingSample]) -> List[Dict]:
            processed_list = []
            for sample in samples_list:
                sample_dict = asdict(sample)
                sample_dict["image"] = str(self.processed_dir / sample_dict.pop("image_path"))
                sample_dict["control"] = str(self.processed_dir / sample_dict.pop("control_path"))
                sample_dict["metadata"] = json.dumps(sample_dict["metadata"])
                processed_list.append(sample_dict)
            return processed_list

        train_samples = process_and_convert_samples(train_samples_raw)
        val_samples = process_and_convert_samples(val_samples_raw)
        test_samples = process_and_convert_samples(test_samples_raw)

        # Convert list of dicts to dict of lists
        train_data = {k: [s[k] for s in train_samples] for k in train_samples[0]} if train_samples else {f: [] for f in features}
        val_data = {k: [s[k] for s in val_samples] for k in val_samples[0]} if val_samples else {f: [] for f in features}
        test_data = {k: [s[k] for s in test_samples] for k in test_samples[0]} if test_samples else {f: [] for f in features}

        train_dataset = Dataset.from_dict(train_data, features=features)
        val_dataset = Dataset.from_dict(val_data, features=features)
        test_dataset = Dataset.from_dict(test_data, features=features)


        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

    def _save_metadata_summary(self, samples: List[TrainingSample]) -> None:
        """Save summary statistics about the dataset."""
        summary = {
            "total_samples": len(samples),
            "unique_patents": len(set(s.patent_id for s in samples)),
            "unique_families": len(self.family_samples),
            "diagram_types": {},
            "cpc_distribution": {},
            "pub_kind_distribution": {},
        }
        
        # Count diagram types
        for sample in samples:
            if sample.diagram_type:
                summary["diagram_types"][sample.diagram_type] = \
                    summary["diagram_types"].get(sample.diagram_type, 0) + 1
        
        # Count CPC codes (first 4 chars only for grouping)
        for sample in samples:
            for cpc in sample.cpc:
                cpc_group = cpc[:4] if len(cpc) >= 4 else cpc
                summary["cpc_distribution"][cpc_group] = \
                    summary["cpc_distribution"].get(cpc_group, 0) + 1
        
        # Count publication kinds
        for sample in samples:
            summary["pub_kind_distribution"][sample.pub_kind] = \
                summary["pub_kind_distribution"].get(sample.pub_kind, 0) + 1
        
        # Save summary
        summary_path = self.processed_dir / "dataset_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Saved dataset summary to %s", summary_path)


def process_uspto_data(config_path: Path) -> DatasetDict:
    """Convenience function for CLI usage."""
    processor = PatentFiguresProcessor.from_config(config_path)
    return processor.process_bulk_downloads()
