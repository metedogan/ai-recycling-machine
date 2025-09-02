"""Inference utilities for TrashNet classification."""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from src.config.settings import CLASS_NAMES, CONFIDENCE_THRESHOLD, MODEL_PATH
from src.data.preprocessor import ImagePreprocessor


class TrashClassifier:
    """Handles inference for trash classification."""
    
    def __init__(self, model_path: Union[str, Path] = MODEL_PATH, 
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = CLASS_NAMES
        self.preprocessor = ImagePreprocessor()
        self.model = None
        
        # Load model if it exists
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Load the trained model."""
        if model_path:
            self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = tf.keras.models.load_model(str(self.model_path))
        print(f"Model loaded from {self.model_path}")
    
    def predict_image(self, image: np.ndarray) -> Dict[str, Union[str, float, np.ndarray]]:
        """Predict the class of a single image."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess_for_inference(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = self.class_names[predicted_class_index]
        
        return {
            'class_name': predicted_class_name,
            'class_index': predicted_class_index,
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist(),
            'meets_threshold': confidence >= self.confidence_threshold
        }
    
    def predict_image_file(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float, np.ndarray]]:
        """Predict the class of an image file."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.predict_image(image)
    
    def predict_batch(self, images: list) -> list:
        """Predict classes for a batch of images."""
        results = []
        for image in images:
            try:
                result = self.predict_image(image)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        return results
    
    def get_top_k_predictions(self, image: np.ndarray, k: int = 3) -> list:
        """Get top-k predictions for an image."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        processed_image = self.preprocessor.preprocess_for_inference(image)
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'class_name': self.class_names[idx],
                'class_index': int(idx),
                'confidence': float(predictions[idx])
            })
        
        return results


class RealTimeClassifier(TrashClassifier):
    """Real-time classification using camera feed."""
    
    def __init__(self, model_path: Union[str, Path] = MODEL_PATH,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 camera_index: int = 0):
        super().__init__(model_path, confidence_threshold)
        self.camera_index = camera_index
        self.cap = None
    
    def start_camera(self) -> bool:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()
    
    def stop_camera(self) -> None:
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def run_real_time_classification(self, display_window: bool = True) -> Optional[Dict]:
        """Run real-time classification loop."""
        if not self.start_camera():
            raise RuntimeError("Could not open camera")
        
        detected_result = None
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Could not read frame from camera")
                    break
                
                # Make prediction
                result = self.predict_image(frame)
                
                if display_window:
                    # Draw prediction on frame
                    text = f"Class: {result['class_name']}, Confidence: {result['confidence']:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow("Real-time Trash Classification", frame)
                
                # Check if confidence threshold is met
                if result['meets_threshold']:
                    detected_result = result
                    print(f"Detected: {result['class_name']} with confidence {result['confidence']:.2f}")
                    break
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.stop_camera()
        
        return detected_result