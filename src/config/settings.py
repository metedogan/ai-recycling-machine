"""Configuration settings for TrashNet classification."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset configuration
DATASET_RESIZED_DIR = DATA_DIR / "dataset-resized"
DATASET_ORIGINAL_DIR = DATA_DIR / "dataset-original"

# Class definitions
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
CLASS_MAPPING = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5
}

# Image dimensions
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
INPUT_SHAPE = (224, 224, 3)  # Standard for MobileNetV2

# Model configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Inference configuration
CONFIDENCE_THRESHOLD = 0.8
MODEL_PATH = PROJECT_ROOT / "model.keras"

# Data split files
TRAIN_LABELS_FILE = DATA_DIR / "one-indexed-files-notrash_train.txt"
TEST_LABELS_FILE = DATA_DIR / "one-indexed-files-notrash_test.txt"
VAL_LABELS_FILE = DATA_DIR / "one-indexed-files-notrash_val.txt"