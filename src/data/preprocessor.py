"""Data preprocessing utilities for TrashNet dataset."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from src.config.settings import IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_SHAPE


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, target_size: Tuple[int, int] = INPUT_SHAPE[:2]):
        self.target_size = target_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    def preprocess_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline for inference."""
        # Resize
        processed = self.resize_image(image)
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        # Normalize
        processed = self.normalize_image(processed)
        return processed
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of images."""
        processed_images = []
        for image in images:
            processed = self.resize_image(image)
            processed = self.normalize_image(processed)
            processed_images.append(processed)
        return np.array(processed_images)


class DatasetResizer:
    """Handles dataset resizing operations."""
    
    def __init__(self, source_dir: Path, dest_dir: Path, 
                 target_height: int = IMAGE_HEIGHT, 
                 target_width: int = IMAGE_WIDTH):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.target_height = target_height
        self.target_width = target_width
    
    def resize_image_file(self, image_path: Path, output_path: Path) -> bool:
        """Resize a single image file."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Rotate if height > width
            height, width = image.shape[:2]
            if height > width:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Resize
            resized = cv2.resize(image, (self.target_width, self.target_height))
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), resized)
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def resize_directory(self, class_name: str) -> int:
        """Resize all images in a class directory."""
        source_class_dir = self.source_dir / class_name
        dest_class_dir = self.dest_dir / class_name
        
        if not source_class_dir.exists():
            print(f"Source directory {source_class_dir} does not exist")
            return 0
        
        processed_count = 0
        for image_file in source_class_dir.glob("*.jpg"):
            output_path = dest_class_dir / image_file.name
            if self.resize_image_file(image_file, output_path):
                processed_count += 1
        
        return processed_count
    
    def resize_all_classes(self, class_names: List[str]) -> Dict[str, int]:
        """Resize images for all classes."""
        results = {}
        for class_name in class_names:
            count = self.resize_directory(class_name)
            results[class_name] = count
            print(f"Processed {count} images for class '{class_name}'")
        return results