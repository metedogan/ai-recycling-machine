"""Model building utilities for TrashNet classification."""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple, Optional
from src.config.settings import INPUT_SHAPE, CLASS_NAMES, LEARNING_RATE


class TrashNetModelBuilder:
    """Builds and configures CNN models for trash classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = INPUT_SHAPE, 
                 num_classes: int = len(CLASS_NAMES)):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_mobilenet_model(self, trainable_base: bool = False) -> Model:
        """Build a model based on MobileNetV2."""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = trainable_base
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def build_custom_cnn(self) -> Model:
        """Build a custom CNN model."""
        model = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: Model, learning_rate: float = LEARNING_RATE) -> Model:
        """Compile the model with optimizer, loss, and metrics."""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_callbacks(self, model_save_path: str, 
                        log_dir: Optional[str] = None) -> list:
        """Create training callbacks."""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if log_dir:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        return callbacks
    
    def get_model_summary(self, model: Model) -> str:
        """Get model architecture summary."""
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()