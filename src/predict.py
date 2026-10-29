import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
from src.preprocess import preprocess_pipeline
from src.main_model import MediaPipeGestureClassifier
from src.data_loader import GestureDataLoader
from src.utils import draw_prediction_overlay

_cached_model = None

def get_trained_model():
    global _cached_model
    if _cached_model is None:
        loader = GestureDataLoader()
        train_imgs, train_lbls, _ = loader.load_split("train")
        if len(train_imgs) == 0:
            train_imgs, train_lbls, _ = loader.load_split("samples")
        _cached_model = MediaPipeGestureClassifier()
        _cached_model.fit(train_imgs, train_lbls)
    return _cached_model

def run_single_prediction(input_path, output_path="outputs/result_prediction.jpg"):
    """Single-command CLI predictor for an input image file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
        
    image = cv2.imread(input_path)
    resized, mask, segmented, blur_score = preprocess_pipeline(image)
    
    model = get_trained_model()
    label, confidence, landmarks = model.predict_single(resized)
    
    annotated = draw_prediction_overlay(resized, label, confidence, blur_score, landmarks)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"Executing single prediction on input image: {input_path}")
    print(f"Predicted Class: {label} | Confidence: {confidence*100:.1f}% | Blur Variance: {blur_score:.1f}")
    print(f"✅ Prediction complete! Output saved to: {output_path}")
    return label, confidence, annotated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-command gesture prediction on an input image.")
    parser.add_argument("--input", required=True, help="Path to input image.")
    parser.add_argument("--output", default="outputs/result_prediction.jpg", help="Path to output annotated image.")
    args = parser.parse_args()
    run_single_prediction(args.input, args.output)
