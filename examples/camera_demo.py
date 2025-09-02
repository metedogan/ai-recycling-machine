#!/usr/bin/env python3
"""
Camera demo for real-time trash classification.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.inference.predictor import RealTimeClassifier
from src.utils.helpers import Logger


def main():
    """Run camera demo."""
    logger = Logger()
    
    # Check if model exists
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.info("Please train a model first using: python scripts/train_model.py")
        return
    
    logger.info("Starting camera demo...")
    logger.info("Press 'q' to quit")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    try:
        # Initialize classifier
        classifier = RealTimeClassifier(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Run real-time classification
        result = classifier.run_real_time_classification(display_window=True)
        
        if result:
            logger.info(f"Final detection: {result['class_name']} "
                       f"(confidence: {result['confidence']:.3f})")
            
            # Show all probabilities
            logger.info("All class probabilities:")
            for i, prob in enumerate(result['all_probabilities']):
                class_name = CLASS_NAMES[i]
                logger.info(f"  {class_name}: {prob:.3f}")
        else:
            logger.info("No confident detection made")
    
    except Exception as e:
        logger.error(f"Camera demo failed: {str(e)}")
        logger.info("Make sure your camera is connected and accessible")


if __name__ == "__main__":
    main()