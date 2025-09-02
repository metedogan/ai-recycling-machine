"""Helper utilities for TrashNet project."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, List
from datetime import datetime


class FileManager:
    """Handles file operations for the project."""
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save data to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
        """Save data to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """Load data from pickle file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str = None) -> Path:
        """Create a timestamped experiment directory."""
        base_dir = Path(base_dir)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        exp_dir = base_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return exp_dir


class MetricsCalculator:
    """Calculates various evaluation metrics."""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                                   num_classes: int) -> Dict[int, float]:
        """Calculate per-class accuracy."""
        per_class_acc = {}
        
        for class_idx in range(num_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
                per_class_acc[class_idx] = class_acc
            else:
                per_class_acc[class_idx] = 0.0
        
        return per_class_acc
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                 num_classes: int) -> np.ndarray:
        """Calculate confusion matrix."""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
        
        return cm


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_file: Union[str, Path] = None):
        self.log_file = Path(log_file) if log_file else None
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\\n")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.log(message, "ERROR")


class ConfigValidator:
    """Validates configuration settings."""
    
    @staticmethod
    def validate_paths(paths: Dict[str, Path]) -> List[str]:
        """Validate that required paths exist."""
        missing_paths = []
        
        for name, path in paths.items():
            if not path.exists():
                missing_paths.append(f"{name}: {path}")
        
        return missing_paths
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        required_keys = ['input_shape', 'num_classes', 'learning_rate']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required config key: {key}")
        
        # Validate input_shape
        if 'input_shape' in config:
            input_shape = config['input_shape']
            if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
                errors.append("input_shape must be a tuple/list of 3 elements (height, width, channels)")
        
        # Validate num_classes
        if 'num_classes' in config:
            if not isinstance(config['num_classes'], int) or config['num_classes'] <= 0:
                errors.append("num_classes must be a positive integer")
        
        # Validate learning_rate
        if 'learning_rate' in config:
            if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
                errors.append("learning_rate must be a positive number")
        
        return errors