"""Two independent gesture heads sharing one genuine MediaPipe hand detector.

- Core: six existing UI labels, optionally from a HaGRID-trained forest.
- Numbers: Number_0 .. Number_5, optionally from a separate HaGRID forest.
- If trained HaGRID files are absent, both heads use *explicitly untrained*
  geometric rules; no fake confidence or accuracy is reported.

Neither model changes the website's HTML, CSS, JS, or six-label API contract.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.geometry_rules import classify_geometry, angle
from src.gesture_runtime import FeatureExtractor, PortableForest, Prediction

CORE_CLASSES = ("Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK")
NUMBER_CLASSES = tuple(f"Number_{i}" for i in range(6))


def read_forest(path: Path, expected_classes):
    """Load only compatible, actually fitted MediaPipe forests."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != "blurry-vision-npz-v1":
            raise ValueError(f"Incompatible forest schema: {path}")
        if str(archive["backend"].item()) != "mediapipe":
            raise ValueError(f"Expected MediaPipe weights (not contour): {path}")
        forest = PortableForest(archive)
    if set(forest.classes_) != set(expected_classes):
        raise ValueError(f"Wrong classes in {path.name}: {forest.classes_}")
    return forest


def _pixel_points(points, image_bgr):
    h, w = image_bgr.shape[:2]
    return np.asarray([[p[0]*w/224, p[1]*h/224] for p in points], dtype=float)


def number_by_geometry(points):
    """Explainable *untrained* numeric-pose estimate; returns None if ambiguous.

    Uses finger joint angles + wrist-to-tip distance. Excludes special signs
    such as OK, rock, and thumbs up so they don't silently become a number.
    This is a fallback only; it must not be called an HaGRID-trained model.
    """
    p = np.asarray(points, dtype=float)
    wrist = p[0]
    palm = max(float(np.linalg.norm(p[9] - wrist)), 1e-6)
    extended = []
    for mcp, pip, tip in ((5,6,8), (9,10,12), (13,14,16), (17,18,20)):
        straight = angle(p[mcp], p[pip], p[tip]) > 152
        far = np.linalg.norm(p[tip]-wrist) > np.linalg.norm(p[pip]-wrist)*1.065
        extended.append(bool(straight and far))
    thumb = (angle(p[1],p[2],p[4]) > 142 and
             np.linalg.norm(p[4]-p[17]) > np.linalg.norm(p[3]-p[17])*1.10)
    pinch = float(np.linalg.norm(p[4]-p[8])/palm) < .43
    index, middle, ring, pinky = extended
    if pinch or (index and pinky and not middle) or (thumb and pinky and not middle):
        return None
    n = sum(extended)
    if n == 0 and thumb:
        return None  # Thumbs Up, not the digit one.
    # '1' and '2' need canonical index and index+middle rather than a rock pose.
    if n == 1 and not index:
        return None
    if n == 2 and not (index and middle):
        return None
    if n >= 3 and not index:
        return None
    return min(n + int(thumb), 5)


class DualRecognizer:
    """Reuse one detector for both tasks; separate weights and evaluation."""
    def __init__(self, core_path, number_path, stream=False):
        self.core_path = Path(core_path)
        self.number_path = Path(number_path)
        self.core_forest = read_forest(self.core_path, CORE_CLASSES)
        self.number_forest = read_forest(self.number_path, NUMBER_CLASSES)
        self.extractor = FeatureExtractor("mediapipe", stream=stream)

    def close(self):
        self.extractor.close()

    def predict_both(self, image_bgr):
        start = time.perf_counter()
        vector, box, points, blur = self.extractor.extract(image_bgr)
        latency = (time.perf_counter()-start)*1000.0
        if vector is None:
            missing = Prediction("No hand detected", 0.0, None, None, blur, latency, "mediapipe")
            return missing, missing
        point_pixels = _pixel_points(points, image_bgr)
        if self.core_forest is not None:
            p = self.core_forest.predict_proba(vector.reshape(1,-1))[0]
            label = self.core_forest.classes_[int(np.argmax(p))]
            prob = float(np.max(p))
            core_backend = "mediapipe-hagrid-six"
        else:
            label, _ = classify_geometry(point_pixels)
            prob = 0.0
            core_backend = "mediapipe-rules"
        core = Prediction(label, prob, box, points, blur, latency, core_backend)
        if self.number_forest is not None:
            p = self.number_forest.predict_proba(vector.reshape(1,-1))[0]
            number_label = self.number_forest.classes_[int(np.argmax(p))]
            number_prob = float(np.max(p))
            number_backend = "mediapipe-hagrid-numbers"
        else:
            n = number_by_geometry(point_pixels)
            number_label = f"Number_{n}" if n is not None else "Not a number gesture"
            number_prob = 0.0
            number_backend = "mediapipe-number-rules"
        numeric = Prediction(number_label, number_prob, box, points, blur, latency,
                             number_backend)
        return core, numeric

    def predict(self, image_bgr):
        return self.predict_both(image_bgr)[0]
