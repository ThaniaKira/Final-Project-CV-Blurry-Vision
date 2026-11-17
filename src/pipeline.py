"""Same application entry point for the demo, tests and image CLI."""
from pathlib import Path
import cv2
from src.gesture_runtime import GestureRecognizer, draw_result, DEFAULT_WEIGHTS

_cached = {}

def get_recognizer(weights=DEFAULT_WEIGHTS, stream=False):
    key = (str(weights), stream)
    if key not in _cached:
        _cached[key] = GestureRecognizer(weights, stream)
    return _cached[key]


def run(image_bgr, weights=DEFAULT_WEIGHTS):
    model = get_recognizer(weights)
    prediction = model.predict(image_bgr)
    return draw_result(image_bgr, prediction), prediction.to_dict()
