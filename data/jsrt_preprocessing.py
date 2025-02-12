"""JSRT Dataset Preprocessing Module

This module handles the preprocessing of JSRT (Japanese Society of Radiological Technology) 
chest X-ray images. It converts the original high-bit depth images to standard 8-bit PNG format
with appropriate normalization and resizing.
"""

import cv2
import logging
import numpy as np
import argparse

from pathlib import Path
from typing import List, Union, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JSRTPreprocessor:
    """Handles preprocessing of JSRT dataset images."""
    
    def __init__(
        self,
        source_dir: Union[str, Path] = "All247images",
        target_size: tuple[int, int] = (1024, 1024),
        original_size: tuple[int, int] = (2048, 2048)
    ):
        """Initialize the preprocessor with configuration.
        
        Args:
            source_dir: Directory containing original JSRT images
            target_size: Output image dimensions (width, height)
            original_size: Original image dimensions (width, height)
        """
        self.source_dir = Path(source_dir)
        self.target_size = target_size
        self.original_size = original_size
        

    def preprocess_files(self, file_list: List[str], output_dir: Union[str, Path], skip_existing: bool = True) -> None:
        """Preprocess a list of JSRT images.
        
        Args:
            file_list: List of image filenames to process
            output_dir: Directory to save processed images
            skip_existing: Skip processing if output file exists
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in file_list:
            self._process_single_file(filename, output_dir, skip_existing)


    def _process_single_file(self, filename: str, output_dir: Path, skip_existing: bool) -> None:
        """Process a single JSRT image file.
        
        Args:
            filename: Name of the image file to process
            output_dir: Directory to save processed image
            skip_existing: Skip processing if output file exists
        """
        input_path = self.source_dir / filename
        output_path = output_dir / filename.replace('.IMG', '.png')
        
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping existing file: {output_path}")
            return
            
        # Read raw image data
        with open(filepath, 'rb') as f:
            dtype = np.dtype('>u2')  # big-endian 16-bit unsigned integer
            img = np.fromfile(f, dtype=dtype).reshape(self.original_size)
        
        # Process image
        img = self._normalize_and_resize(img)
        
        # Save processed image
        cv2.imwrite(str(output_path), img)
        logger.info(f"Processed and saved: {output_path}")
        

    def _normalize_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Normalize and resize the image."""
        # Normalize to [0, 1] range and invert
        img = 1 - img.astype('float32') / 4096
        
        # Resize to target dimensions
        img = cv2.resize(img, self.target_size)
        
        # Scale to [0, 255] range and convert to 8-bit
        img = (img * 255).astype('uint8')
        return img


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess JSRT dataset images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source-dir', type=str, default='All247images', help='Directory containing original JSRT images')
    
    parser.add_argument('--train-files', type=str, default='train_files.txt', help='Path to file containing training image filenames')
    parser.add_argument('--val-files', type=str, default='val_files.txt', help='Path to file containing validation image filenames')
    parser.add_argument('--test-files', type=str, default='test_files.txt', help='Path to file containing test image filenames')
    
    parser.add_argument('--output-dir', type=str, default='.', help='Base directory for output images')
    
    parser.add_argument('--target-size', type=int, nargs=2, default=[1024, 1024], help='Target image size as width height (e.g., 1024 1024)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip processing existing files')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    preprocessor = JSRTPreprocessor(
        source_dir=args.source_dir,
        target_size=tuple(args.target_size)  # Convert list to tuple
    )
    
    output_base = Path(args.output_dir)
    train_output = output_base / "Train/Images"
    val_output = output_base / "Val/Images"
    test_output = output_base / "Test/Images"
    
    # Process training set
    with open(args.train_files, 'r') as f:
        train_files = f.read().splitlines()
    preprocessor.preprocess_files(train_files, train_output, args.skip_existing)
    logger.info("Training images preprocessed")
    
    # Process validation set
    with open(args.val_files, 'r') as f:
        val_files = f.read().splitlines()
    preprocessor.preprocess_files(val_files, val_output, args.skip_existing)
    logger.info("Validation images preprocessed")
    
    # Process test set
    with open(args.test_files, 'r') as f:
        test_files = f.read().splitlines()
    preprocessor.preprocess_files(test_files, test_output, args.skip_existing)
    logger.info("Test images preprocessed")

if __name__ == "__main__":
    main()