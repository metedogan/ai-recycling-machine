"""Model training utilities for TrashNet classification."""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.config.settings import BATCH_SIZE, EPOCHS, MODELS_DIR, LOGS_DIR
from src.data.preprocessor import ImagePreprocessor
from src.models.model_builder import TrashNetModelBuilder


class ModelTrainer:
    """Handles model training operations."""
    
    def __init__(self, model_builder: TrashNetModelBuilder):
        self.model_builder = model_builder
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.history = None
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data."""
        # Preprocess images
        X_train_processed = self.preprocessor.preprocess_batch(X_train)
        X_val_processed = self.preprocessor.preprocess_batch(X_val)
        
        return X_train_processed, y_train, X_val_processed, y_val
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              batch_size: int = BATCH_SIZE) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow data generators with augmentation."""
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_type: str = 'mobilenet',
                   epochs: int = EPOCHS,
                   batch_size: int = BATCH_SIZE,
                   use_data_augmentation: bool = True,
                   model_save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the model."""
        
        # Create model
        if model_type == 'mobilenet':
            self.model = self.model_builder.build_mobilenet_model()
        elif model_type == 'custom':
            self.model = self.model_builder.build_custom_cnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        self.model = self.model_builder.compile_model(self.model)
        
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep = self.prepare_data(
            X_train, y_train, X_val, y_val
        )
        
        # Set up paths
        if model_save_path is None:
            MODELS_DIR.mkdir(exist_ok=True)
            model_save_path = str(MODELS_DIR / f"trashnet_{model_type}_best.keras")
        
        LOGS_DIR.mkdir(exist_ok=True)
        log_dir = str(LOGS_DIR / f"trashnet_{model_type}")
        
        # Create callbacks
        callbacks = self.model_builder.create_callbacks(model_save_path, log_dir)
        
        # Train model
        if use_data_augmentation:
            train_gen, val_gen = self.create_data_generators(
                X_train_prep, y_train_prep, X_val_prep, y_val_prep, batch_size
            )
            
            steps_per_epoch = len(X_train_prep) // batch_size
            validation_steps = len(X_val_prep) // batch_size
            
            self.history = self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train_prep, y_train_prep,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val_prep, y_val_prep),
                callbacks=callbacks,
                verbose=1
            )
        
        return {
            'model': self.model,
            'history': self.history,
            'model_path': model_save_path
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        X_test_prep = self.preprocessor.preprocess_batch(X_test)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test_prep, y_test, verbose=0)
        
        # Get predictions for detailed metrics
        y_pred = self.model.predict(X_test_prep, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate per-class accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }