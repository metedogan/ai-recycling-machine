#!/usr/bin/env python3
"""
Data preprocessing script for TrashNet dataset.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.data.preprocessor import DatasetResizer
from src.utils.helpers import Logger


def main():
    parser = argparse.ArgumentParser(description='Preprocess TrashNet dataset')
    parser.add_argument('--source-dir', type=str, default=str(DATASET_ORIGINAL_DIR),
                       help='Source directory containing original images')
    parser.add_argument('--dest-dir', type=str, default=str(DATASET_RESIZED_DIR),
                       help='Destination directory for resized images')
    parser.add_argument('--height', type=int, default=IMAGE_HEIGHT,
                       help='Target image height')
    parser.add_argument('--width', type=int, default=IMAGE_WIDTH,
                       help='Target image width')
    
    args = parser.parse_args()
    
    logger = Logger()
    
    logger.info("Starting data preprocessing...")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Destination directory: {args.dest_dir}")
    logger.info(f"Target dimensions: {args.width}x{args.height}")
    
    try:
        # Initialize resizer
        resizer = DatasetResizer(
            source_dir=args.source_dir,
            dest_dir=args.dest_dir,
            target_height=args.height,
            target_width=args.width
        )
        
        # Check if source directory exists
        if not Path(args.source_dir).exists():
            logger.error(f"Source directory does not exist: {args.source_dir}")
            return
        
        # Resize all classes
        results = resizer.resize_all_classes(CLASS_NAMES)
        
        # Summary
        total_processed = sum(results.values())
        logger.info(f"Preprocessing completed!")
        logger.info(f"Total images processed: {total_processed}")
        
        for class_name, count in results.items():
            logger.info(f"  {class_name}: {count} images")
        
        if total_processed == 0:
            logger.warning("No images were processed. Check source directory structure.")
            logger.info("Expected structure:")
            logger.info("  source_dir/")
            for class_name in CLASS_NAMES:
                logger.info(f"    {class_name}/")
                logger.info(f"      *.jpg")
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()