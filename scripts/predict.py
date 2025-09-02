#!/usr/bin/env python3
"""
Prediction script for TrashNet classification.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.inference.predictor import TrashClassifier, RealTimeClassifier
from src.utils.helpers import Logger


def main():
    parser = argparse.ArgumentParser(description='Predict trash class using trained model')
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATH),
                       help='Path to trained model')
    parser.add_argument('--image-path', type=str,
                       help='Path to image file for prediction')
    parser.add_argument('--camera', action='store_true',
                       help='Use camera for real-time prediction')
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for predictions')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Show top-k predictions')
    
    args = parser.parse_args()
    
    logger = Logger()
    
    try:
        if args.camera:
            # Real-time classification
            logger.info("Starting real-time classification...")
            classifier = RealTimeClassifier(
                model_path=args.model_path,
                confidence_threshold=args.confidence_threshold
            )
            
            result = classifier.run_real_time_classification()
            
            if result:
                logger.info(f"Final detection: {result['class_name']} "
                          f"(confidence: {result['confidence']:.3f})")
            else:
                logger.info("No detection above threshold")
        
        elif args.image_path:
            # Single image prediction
            logger.info(f"Predicting class for image: {args.image_path}")
            
            classifier = TrashClassifier(
                model_path=args.model_path,
                confidence_threshold=args.confidence_threshold
            )
            
            # Get top-k predictions
            import cv2
            image = cv2.imread(args.image_path)
            if image is None:
                logger.error(f"Could not load image: {args.image_path}")
                return
            
            top_predictions = classifier.get_top_k_predictions(image, args.top_k)
            
            logger.info("Top predictions:")
            for i, pred in enumerate(top_predictions, 1):
                logger.info(f"{i}. {pred['class_name']}: {pred['confidence']:.3f}")
            
            # Check if meets threshold
            best_pred = top_predictions[0]
            if best_pred['confidence'] >= args.confidence_threshold:
                logger.info(f"✓ Confident prediction: {best_pred['class_name']}")
            else:
                logger.info(f"⚠ Low confidence prediction (threshold: {args.confidence_threshold})")
        
        else:
            logger.error("Please specify either --image-path or --camera")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()