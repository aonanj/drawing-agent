from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TIFFImageProcessor:
    """Process USPTO patent TIFF images for training."""

    def __init__(
        self,
        target_size: int = 1024,
        min_gap: int = 40,
        gap_threshold: float = 0.005,
        dpi: int = 300,
    ):
        self.target_size = target_size
        self.min_gap = min_gap
        self.gap_threshold = gap_threshold
        self.dpi = dpi

    def process_tiff(
        self,
        tiff_path: Path,
        output_dir: Path,
        patent_id: str,
    ) -> List[Tuple[Path, Path]]:
        """
        Process a TIFF file, splitting multi-figure images and generating control maps.
        
        Returns:
            List of (target_image_path, control_map_path) tuples
        """
        try:
            # Load TIFF image
            image = Image.open(tiff_path)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Handle multi-page TIFFs
            results = []
            page_num = 0
            
            try:
                while True:
                    image.seek(page_num)
                    img_array = np.array(image.convert("L"))  # Convert to grayscale
                    
                    # Process this page
                    page_results = self._process_single_page(
                        img_array, output_dir, patent_id, page_num
                    )
                    results.extend(page_results)
                    
                    page_num += 1
            except EOFError:
                # No more pages
                pass
            
            return results
            
        except Exception as e:
            logger.error("Failed to process TIFF %s: %s", tiff_path, e)
            return []

    def _process_single_page(
        self,
        img_array: np.ndarray,
        output_dir: Path,
        patent_id: str,
        page_num: int,
    ) -> List[Tuple[Path, Path]]:
        """Process a single page from a TIFF."""
        # Binarize the image
        img_binary = self._binarize(img_array)
        
        # Deskew if needed
        img_deskewed = self._deskew(img_binary)
        
        # Denoise
        img_clean = self._denoise(img_deskewed)
        
        # Detect and split figures
        figure_crops = self._split_figures(img_clean)
        
        if not figure_crops:
            # No clear figures detected, treat whole page as one figure
            figure_crops = [img_clean]
        
        # Process each figure
        results = []
        for fig_idx, fig_crop in enumerate(figure_crops):
            # Generate figure identifier
            if len(figure_crops) == 1 and page_num == 0:
                fig_id = f"{patent_id}"
            else:
                fig_id = f"{patent_id}_p{page_num}_f{fig_idx}"
            
            # Prepare final image
            img_final = self._prepare_final_image(fig_crop)
            
            # Generate control map
            control_map = self._generate_control_map(img_final)
            
            # Save images
            target_path = output_dir / f"{fig_id}.png"
            control_path = output_dir / f"{fig_id}_canny.png"
            
            try:
                Image.fromarray(img_final).save(target_path)
                Image.fromarray(control_map).save(control_path)
                results.append((target_path, control_path))
            except Exception as e:
                logger.error("Failed to save images for %s: %s", fig_id, e)
        
        return results

    def _binarize(self, img_array: np.ndarray) -> np.ndarray:
        """Binarize image using Otsu's method."""
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Apply Otsu's thresholding
        _, img_binary = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Invert if background is dark
        if img_binary.mean() < 128:
            img_binary = cv2.bitwise_not(img_binary)
        
        return img_binary

    def _deskew(self, img_binary: np.ndarray) -> np.ndarray:
        """Deskew the image."""
        # Find all non-zero points
        coords = np.column_stack(np.where(img_binary < 128))
        
        if len(coords) < 100:
            return img_binary
        
        # Calculate angle using minAreaRect
        try:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only deskew if angle is significant
            if abs(angle) < 0.5:
                return img_binary
            
            # Rotate image
            h, w = img_binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_deskewed = cv2.warpAffine(
                img_binary,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            
            return img_deskewed
        except Exception as e:
            logger.debug("Deskewing failed: %s", e)
            return img_binary

    def _denoise(self, img_binary: np.ndarray) -> np.ndarray:
        """Remove small noise artifacts."""
        # Morphological opening to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        img_opened = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        
        # Remove very small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(img_opened), connectivity=8
        )
        
        # Create mask for components to keep
        min_size = 50  # Minimum component size in pixels
        mask = np.zeros_like(img_opened)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        # Apply mask
        img_clean = cv2.bitwise_or(img_opened, cv2.bitwise_not(mask))
        
        return img_clean

    def _split_figures(self, img_binary: np.ndarray) -> List[np.ndarray]:
        """
        Split multi-figure images by detecting distinct figure groups.
        Uses connected component analysis to avoid breaking up flowcharts
        and other multi-part diagrams.

        Returns:
            List of cropped figure images
        """
        h, w = img_binary.shape

        # Invert for connected components (find black content)
        img_inverted = cv2.bitwise_not(img_binary)

        # Apply morphological closing to connect nearby components
        # This helps keep flowchart boxes and connected diagrams together
        kernel = np.ones((15, 15), np.uint8)
        img_closed = cv2.morphologyEx(img_inverted, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_closed, connectivity=8
        )

        # Filter components by size and characteristics
        valid_components = []
        min_area = (h * w) * 0.01  # At least 1% of image
        min_dimension = 250  # Minimum width or height

        for i in range(1, num_labels):  # Skip background (0)
            x, y, comp_w, comp_h, area = stats[i]

            # Skip tiny components
            if area < min_area:
                continue

            # Skip very small regions (likely just text labels)
            if comp_w < min_dimension or comp_h < min_dimension:
                continue

            # Extract the component region
            component_mask = (labels == i).astype(np.uint8) * 255
            component_region = cv2.bitwise_and(img_binary, img_binary, mask=component_mask)

            # Check if it's a text-only region (typically very wide and short)
            # or has very little actual content complexity
            if self._is_text_label_region(component_region[y:y+comp_h, x:x+comp_w]):
                logger.debug("Filtering out text label region at (%d, %d, %d, %d)",
                           x, y, comp_w, comp_h)
                continue

            valid_components.append((y, y + comp_h, x, x + comp_w))

        # If no valid components found, return empty list (whole image is one figure)
        if not valid_components:
            return []

        # If only one component, also return empty (whole image is one figure)
        if len(valid_components) == 1:
            return []

        # Sort components by vertical position
        valid_components.sort(key=lambda c: c[0])

        # Check if components should be merged based on proximity
        merged_components = self._merge_nearby_components(valid_components, h, w)

        # If merging resulted in single component, treat whole image as one figure
        if len(merged_components) <= 1:
            return []

        # Extract figure crops
        figures = []
        for y_start, y_end, x_start, x_end in merged_components:
            # Add some padding
            y_start = max(0, y_start - 20)
            y_end = min(h, y_end + 20)
            x_start = max(0, x_start - 20)
            x_end = min(w, x_end + 20)

            fig_crop = img_binary[y_start:y_end, x_start:x_end]
            figures.append(fig_crop)

        return figures

    def _is_text_label_region(self, region: np.ndarray) -> bool:
        """
        Detect if a region is likely just a text label (e.g., "FIG. 3").

        Args:
            region: Binary image region to check

        Returns:
            True if region appears to be just text
        """
        h, w = region.shape

        if h == 0 or w == 0:
            return True

        # Text labels are typically very wide and short
        aspect_ratio = w / h
        if aspect_ratio > 8 or aspect_ratio < 0.125:
            # Very extreme aspect ratio suggests text label
            return True

        # Check content density - text usually has lower density than diagrams
        content_pixels = np.sum(region < 128)
        total_pixels = h * w
        density = content_pixels / total_pixels if total_pixels > 0 else 0

        # Text labels typically have very low content density (< 5%)
        if density < 0.05:
            return True

        # Check height - figure labels are usually quite short (< 150 pixels)
        if h < 150 and aspect_ratio > 3:
            return True

        return False

    def _merge_nearby_components(
        self,
        components: List[Tuple[int, int, int, int]],
        _img_h: int,
        _img_w: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Merge components that are close together vertically.
        This helps keep multi-part diagrams (like flowcharts) together.

        Args:
            components: List of (y_start, y_end, x_start, x_end) tuples
            _img_h: Image height (unused, kept for API consistency)
            _img_w: Image width (unused, kept for API consistency)

        Returns:
            Merged list of components
        """
        if len(components) <= 1:
            return components

        merged = []
        current = list(components[0])  # [y_start, y_end, x_start, x_end]

        for i in range(1, len(components)):
            next_comp = components[i]

            # Calculate gap between current and next component
            gap = next_comp[0] - current[1]

            # Merge if gap is small (less than min_gap threshold)
            # This keeps flowchart boxes and related diagram parts together
            if gap < self.min_gap:
                # Merge: extend current component
                current[1] = max(current[1], next_comp[1])  # y_end
                current[0] = min(current[0], next_comp[0])  # y_start
                current[2] = min(current[2], next_comp[2])  # x_start
                current[3] = max(current[3], next_comp[3])  # x_end
            else:
                # Gap is large enough - this is a separate figure
                merged.append(tuple(current))
                current = list(next_comp)

        # Add the last component
        merged.append(tuple(current))

        return merged

    def _prepare_final_image(self, img_crop: np.ndarray) -> np.ndarray:
        """
        Prepare final image: crop content, pad to square, resize to target size.
        """
        # Find content boundaries
        rows_with_content = np.any(img_crop < 250, axis=1)
        cols_with_content = np.any(img_crop < 250, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            # Empty image, return white square
            return np.full((self.target_size, self.target_size), 255, dtype=np.uint8)
        
        row_start = np.argmax(rows_with_content)
        row_end = len(rows_with_content) - np.argmax(rows_with_content[::-1])
        col_start = np.argmax(cols_with_content)
        col_end = len(cols_with_content) - np.argmax(cols_with_content[::-1])
        
        # Crop to content with small margin
        margin = 20
        row_start = max(0, row_start - margin)
        row_end = min(img_crop.shape[0], row_end + margin)
        col_start = max(0, col_start - margin)
        col_end = min(img_crop.shape[1], col_end + margin)
        
        img_cropped = img_crop[row_start:row_end, col_start:col_end]
        
        # Pad to square
        h, w = img_cropped.shape
        max_dim = max(h, w)
        
        # Create white square
        img_square = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        
        # Center the image
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        img_square[y_offset:y_offset + h, x_offset:x_offset + w] = img_cropped
        
        # Resize to target size
        img_resized = cv2.resize(
            img_square,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        
        return img_resized

    def _generate_control_map(self, img_binary: np.ndarray) -> np.ndarray:
        """Generate Canny edge map for ControlNet."""
        # Apply Gaussian blur to reduce noise
        img_blurred = cv2.GaussianBlur(img_binary, (3, 3), 0)
        
        # Apply Canny edge detection
        # For binary line art, use tight thresholds
        edges = cv2.Canny(img_blurred, threshold1=100, threshold2=200)
        
        # Dilate edges slightly to make them more visible
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges


def validate_image_quality(img_path: Path) -> bool:
    """
    Validate that an image meets quality requirements.
    
    Returns:
        True if image passes quality checks
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Check resolution
        h, w = img_array.shape[:2]
        if min(h, w) < 512:
            logger.warning("Image %s resolution too low: %dx%d", img_path, w, h)
            return False
        
        # Check if grayscale/monochrome
        if len(img_array.shape) == 3 and img_array.shape[2] > 1:
            # Check if it's effectively grayscale
            if not np.allclose(img_array[:, :, 0], img_array[:, :, 1], atol=5):
                logger.warning("Image %s is not monochrome", img_path)
                return False
        
        # Check if mostly blank
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = img_array[:, :, 0]
        
        content_ratio = np.sum(gray < 250) / gray.size
        if content_ratio < 0.01:
            logger.warning("Image %s has too little content: %.2f%%", img_path, content_ratio * 100)
            return False
        
        return True
        
    except Exception as e:
        logger.error("Failed to validate image %s: %s", img_path, e)
        return False