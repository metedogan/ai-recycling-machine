#!/usr/bin/env python3
"""
Basic usage examples for TrashNet classification.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.data.loader import DatasetLoader
from src.data.preprocessor import ImagePreprocessor
from src.models.model_builder import TrashNetModelBuilder
from src.models.trainer import ModelTrainer
from src.inference.predictor import TrashClassifier
from src.utils.visualization import DatasetVisualizer, TrainingVisualizer
from src.utils.helpers import Logger


def example_data_loading():
    """Example: Load and explore dataset."""
    print("=== Data Loading Example ===")
    
    loader = DatasetLoader()
    
    # Load full dataset
    X, y = loader.load_full_dataset()
    print(f"Loaded {len(X)} images")
    
    # Show class distribution
    distribution = loader.get_class_distribution(y)
    print("Class distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count}")
    
    # Visualize samples
    visualizer = DatasetVisualizer()
    visualizer.show_sample_images(X[:50], y[:50], CLASS_NAMES, samples_per_class=2)


def example_model_building():
    """Example: Build and compile models."""
    print("=== Model Building Example ===")
    
    builder = TrashNetModelBuilder()
    
    # Build MobileNet model
    model = builder.build_mobilenet_model()
    model = builder.compile_model(model)
    
    print("MobileNet model summary:")
    print(builder.get_model_summary(model))
    
    # Build custom CNN
    custom_model = builder.build_custom_cnn()
    custom_model = builder.compile_model(custom_model)
    
    print("Custom CNN model summary:")
    print(builder.get_model_summary(custom_model))


def example_training():
    """Example: Train a model (small scale for demo)."""
    print("=== Training Example ===")
    
    # Load small subset for quick demo
    loader = DatasetLoader()
    X, y = loader.load_full_dataset()
    
    # Use only first 100 samples for quick demo
    X_small = X[:100]
    y_small = y[:100]
    
    # Split data
    X_train, X_val, y_train, y_val = loader.create_train_val_split(
        X_small, y_small, test_size=0.3
    )
    
    print(f"Training on {len(X_train)} samples")
    print(f"Validating on {len(X_val)} samples")
    
    # Build and train model
    builder = TrashNetModelBuilder()
    trainer = ModelTrainer(builder)
    
    results = trainer.train_model(
        X_train, y_train, X_val, y_val,
        model_type='custom',
        epochs=5,  # Small number for demo
        batch_size=16,
        use_data_augmentation=False
    )
    
    print("Training completed!")
    print(f"Final validation accuracy: {results['history'].history['val_accuracy'][-1]:.3f}")


def example_inference():
    """Example: Use trained model for inference."""
    print("=== Inference Example ===")
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please train a model first using scripts/train_model.py")
        return
    
    # Load classifier
    classifier = TrashClassifier()
    
    # Load a sample image
    loader = DatasetLoader()
    X, y = loader.load_full_dataset()
    
    if len(X) == 0:
        print("No images found in dataset")
        return
    
    # Predict on first image
    sample_image = X[0]
    true_class = CLASS_NAMES[y[0]]
    
    result = classifier.predict_image(sample_image)
    
    print(f"True class: {true_class}")
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Meets threshold: {result['meets_threshold']}")
    
    # Get top-3 predictions
    top_predictions = classifier.get_top_k_predictions(sample_image, k=3)
    print("Top 3 predictions:")
    for i, pred in enumerate(top_predictions, 1):
        print(f"  {i}. {pred['class_name']}: {pred['confidence']:.3f}")


def example_preprocessing():
    """Example: Image preprocessing."""
    print("=== Preprocessing Example ===")
    
    preprocessor = ImagePreprocessor()
    
    # Load a sample image
    loader = DatasetLoader()
    X, y = loader.load_full_dataset()
    
    if len(X) == 0:
        print("No images found in dataset")
        return
    
    sample_image = X[0]
    print(f"Original image shape: {sample_image.shape}")
    
    # Preprocess for inference
    processed = preprocessor.preprocess_for_inference(sample_image)
    print(f"Processed image shape: {processed.shape}")
    print(f"Pixel value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Preprocess batch
    batch = X[:5]
    processed_batch = preprocessor.preprocess_batch(batch)
    print(f"Processed batch shape: {processed_batch.shape}")


def main():
    """Run all examples."""
    logger = Logger()
    
    examples = [
        ("Data Loading", example_data_loading),
        ("Model Building", example_model_building),
        ("Preprocessing", example_preprocessing),
        ("Inference", example_inference),
        # ("Training", example_training),  # Commented out as it takes time
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*50}")
            print(f"Running {name} Example")
            print(f"{'='*50}")
            example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Examples completed!")
    print("To run training example, uncomment it in the examples list.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()