#!/usr/bin/env python3
"""
Training script for TrashNet classification model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.data.loader import DatasetLoader
from src.models.model_builder import TrashNetModelBuilder
from src.models.trainer import ModelTrainer
from src.utils.helpers import Logger, FileManager
from src.utils.visualization import TrainingVisualizer


def main():
    parser = argparse.ArgumentParser(description='Train TrashNet classification model')
    parser.add_argument('--model-type', choices=['mobilenet', 'custom'], 
                       default='mobilenet', help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--output-dir', type=str, default=str(MODELS_DIR),
                       help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output_dir) / "training.log"
    logger = Logger(log_file)
    
    logger.info("Starting TrashNet model training")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        loader = DatasetLoader()
        
        # Check if we have split files
        if TRAIN_LABELS_FILE.exists() and VAL_LABELS_FILE.exists():
            logger.info("Using predefined train/val split")
            
            # Load labels
            train_labels = loader.load_labels_from_file(TRAIN_LABELS_FILE)
            val_labels = loader.load_labels_from_file(VAL_LABELS_FILE)
            
            # Load images
            X_train, y_train = loader.load_images_and_labels(
                list(train_labels.keys()), train_labels
            )
            X_val, y_val = loader.load_images_and_labels(
                list(val_labels.keys()), val_labels
            )
        else:
            logger.info("Creating train/val split from full dataset")
            
            # Load full dataset
            X_full, y_full = loader.load_full_dataset()
            
            # Split dataset
            X_train, X_val, y_train, y_val = loader.create_train_val_split(
                X_full, y_full, test_size=VALIDATION_SPLIT
            )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Show class distribution
        train_dist = loader.get_class_distribution(y_train)
        val_dist = loader.get_class_distribution(y_val)
        
        logger.info("Training class distribution:")
        for class_name, count in train_dist.items():
            logger.info(f"  {class_name}: {count}")
        
        # Initialize model builder and trainer
        model_builder = TrashNetModelBuilder()
        trainer = ModelTrainer(model_builder)
        
        # Train model
        logger.info("Starting model training...")
        
        model_save_path = Path(args.output_dir) / f"trashnet_{args.model_type}_best.keras"
        
        results = trainer.train_model(
            X_train, y_train, X_val, y_val,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_data_augmentation=not args.no_augmentation,
            model_save_path=str(model_save_path)
        )
        
        logger.info("Training completed!")
        logger.info(f"Model saved to: {results['model_path']}")
        
        # Save training history
        history_path = Path(args.output_dir) / f"training_history_{args.model_type}.json"
        FileManager.save_json(results['history'].history, history_path)
        logger.info(f"Training history saved to: {history_path}")
        
        # Create visualizations
        logger.info("Creating training visualizations...")
        visualizer = TrainingVisualizer()
        
        # Plot training history
        plot_path = Path(args.output_dir) / f"training_plot_{args.model_type}.png"
        visualizer.plot_training_history(results['history'].history, str(plot_path))
        
        # Evaluate on validation set
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate_model(X_val, y_val)
        
        logger.info(f"Validation accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"Validation loss: {eval_results['loss']:.4f}")
        
        # Save evaluation results
        eval_path = Path(args.output_dir) / f"evaluation_results_{args.model_type}.json"
        FileManager.save_json(eval_results, eval_path)
        
        # Plot confusion matrix
        cm_path = Path(args.output_dir) / f"confusion_matrix_{args.model_type}.png"
        visualizer.plot_confusion_matrix(
            np.array(eval_results['confusion_matrix']), 
            CLASS_NAMES, 
            str(cm_path)
        )
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()