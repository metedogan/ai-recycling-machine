"""Data loading utilities for TrashNet dataset."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from src.config.settings import CLASS_MAPPING, CLASS_NAMES, DATASET_RESIZED_DIR


class DatasetLoader:
    """Handles loading and splitting of TrashNet dataset."""
    
    def __init__(self, dataset_dir: Path = DATASET_RESIZED_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.class_mapping = CLASS_MAPPING
        self.class_names = CLASS_NAMES
    
    def load_labels_from_file(self, labels_file: Path) -> Dict[str, int]:
        """Load labels from text file."""
        labels = {}
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = int(parts[1]) - 1  # Convert from 1-indexed to 0-indexed
                        labels[filename] = label
        return labels
    
    def load_images_and_labels(self, image_files: List[str], 
                              labels_dict: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load images and their corresponding labels."""
        images = []
        labels = []
        
        for filename in image_files:
            # Try to find the image in any class directory
            image_path = None
            for class_name in self.class_names:
                potential_path = self.dataset_dir / class_name / filename
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            images.append(image)
            
            # Get label
            if labels_dict and filename in labels_dict:
                labels.append(labels_dict[filename])
            else:
                # Infer label from directory name
                class_name = image_path.parent.name
                labels.append(self.class_mapping.get(class_name, 0))
        
        return np.array(images), np.array(labels)
    
    def load_class_directory(self, class_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all images from a specific class directory."""
        class_dir = self.dataset_dir / class_name
        if not class_dir.exists():
            return np.array([]), np.array([])
        
        images = []
        labels = []
        class_label = self.class_mapping[class_name]
        
        for image_file in class_dir.glob("*.jpg"):
            image = cv2.imread(str(image_file))
            if image is not None:
                images.append(image)
                labels.append(class_label)
        
        return np.array(images), np.array(labels)
    
    def load_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the complete dataset from all class directories."""
        all_images = []
        all_labels = []
        
        for class_name in self.class_names:
            images, labels = self.load_class_directory(class_name)
            if len(images) > 0:
                all_images.extend(images)
                all_labels.extend(labels)
        
        return np.array(all_images), np.array(all_labels)
    
    def create_train_val_split(self, images: np.ndarray, labels: np.ndarray, 
                              test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset into training and validation sets."""
        return train_test_split(images, labels, test_size=test_size, 
                              random_state=random_state, stratify=labels)
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}
        for class_idx, count in zip(unique, counts):
            class_name = self.class_names[class_idx]
            distribution[class_name] = count
        return distribution