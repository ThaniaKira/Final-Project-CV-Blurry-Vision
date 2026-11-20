"""Shared model, image processing and rendering for CLI, UI and evaluation.

The two explicit feature backends are intentionally distinct: `contour` uses
classical OpenCV features; `mediapipe` requires the actual MediaPipe library.
No synthetic/fake landmarks are ever reported as MediaPipe detections.
"""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import preprocess_pipeline, compute_blur_score

CLASSES = ("Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK")
DEFAULT_WEIGHTS = Path("models/gesture_contour.npz")


class BackendUnavailable(RuntimeError):
    pass


class FeatureExtractor:
    """Extract a real hand region and a fixed-length feature vector."""

    def __init__(self, backend: str = "contour", stream: bool = False):
        if backend not in ("contour", "mediapipe"):
            raise ValueError("backend must be 'contour' or 'mediapipe'")
        self.backend = backend
        self.hands = None
        if backend == "mediapipe":
            try:
                import mediapipe as mp
                if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
                    raise BackendUnavailable("MediaPipe solutions.hands is unavailable; install the supported version")
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=not stream, max_num_hands=1,
                    # Snapshot uploads frequently have uneven room lighting.
                    # 0.35 retains MediaPipe's landmark validation while avoiding
                    # unnecessary missed hands from the 0.50 demo default.
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.40,
                    model_complexity=1,
                )
            except ImportError as exc:
                raise BackendUnavailable("MediaPipe is not installed; use --backend contour or install requirements.txt") from exc

    def close(self):
        if self.hands is not None:
            self.hands.close()

    def _detect_hand_landmarks(self, frame_bgr):
        """Run a normal pass, then a low-light-safe pass if necessary.

        The second pass changes only luminance, so any coordinates still refer
        exactly to the source frame.  It avoids the misleading alternative of
        drawing estimated fingers when MediaPipe has not detected a real hand.
        """
        result = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            return result
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
        return self.hands.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    def extract(self, image_bgr):
        """Return (feature or None, box or None, points or None, blur_score).

        Box is (x1,y1,x2,y2) in processed 224x224 pixels; points are (x,y).
        No hand -> feature=None; never generate placeholder points.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty or unreadable image")
        # Preserve original aspect ratio, resolution and color for MediaPipe;
        # the classical skin detector alone uses a 224x224 blurred/thresholded input.
        if self.backend == "mediapipe":
            height, width = image_bgr.shape[:2]
            blur_score = compute_blur_score(cv2.resize(image_bgr, (224, 224)))
            scale = min(1.0, 960.0/max(height,width))
            frame = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale<1 else image_bgr
            result = self._detect_hand_landmarks(frame)
            if not result.multi_hand_landmarks:
                return None, None, None, blur_score
            points = np.asarray([[p.x,p.y,p.z] for p in result.multi_hand_landmarks[0].landmark], dtype=np.float32)
            # Use pixel coordinates before normalizing. Raw MediaPipe x/y values
            # represent relative image width/height, not equal-length units.
            xy_pixel = points[:,:2]*np.asarray([width,height],dtype=np.float32)
            center = xy_pixel[0]
            palm_span = max(float(np.linalg.norm(xy_pixel[9]-center)), 1.0)
            relative_xy = (xy_pixel-center)/palm_span
            relative_z = ((points[:,2]-points[0,2])*width/palm_span).reshape(-1,1)
            feature = np.hstack((relative_xy,relative_z)).ravel().astype(np.float32)
            xy = np.clip(points[:,:2]*224,0,223).astype(int)
            x1,y1 = xy.min(axis=0).tolist()
            x2,y2 = xy.max(axis=0).tolist()
            # Landmarks sit inside the fingers, so include a small visual margin
            # rather than returning a box that appears to cut fingertips off.
            margin = 8
            x1, y1 = max(0, x1-margin), max(0, y1-margin)
            x2, y2 = min(223, x2+margin), min(223, y2+margin)
            return feature, (x1,y1,x2,y2), xy.tolist(), blur_score

        resized, mask, _, _ = preprocess_pipeline(image_bgr, target_size=(224,224))
        blur_score = compute_blur_score(cv2.resize(image_bgr, (224,224)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, blur_score
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 224 * 224 * 0.012:
            return None, None, None, blur_score
        x, y, w, h = cv2.boundingRect(contour)
        # Classical region shape/spatial features (not anatomical landmarks).
        silhouette = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(silhouette, [contour], -1, 255, thickness=-1)
        roi = silhouette[y:y+h, x:x+w]
        normalized = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        moments = cv2.HuMoments(cv2.moments(contour)).ravel()
        moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-12)
        hull = cv2.convexHull(contour)
        hull_area = max(cv2.contourArea(hull), 1.0)
        shape = np.array([area / (224*224), w/224, h/224, w/max(h, 1),
                          area/hull_area, cv2.arcLength(contour, True)/(2*224),
                          x/224, y/224], dtype=np.float32)
        feat = np.hstack((normalized.ravel(), np.clip(moments, -12, 12), shape)).astype(np.float32)
        return feat, (x, y, x+w, y+h), None, blur_score


@dataclass
class Prediction:
    label: str
    confidence: float
    box: tuple | None
    points: list | None
    blur_score: float
    latency_ms: float
    backend: str

    def to_dict(self):
        return {"gesture": self.label, "confidence": round(self.confidence, 4),
                "hand_count": int(self.box is not None), "bounding_box": self.box,
                "blur_variance": round(self.blur_score, 2),
                "processing_ms": round(self.latency_ms, 2),
                "estimated_fps": round(1000/self.latency_ms, 2) if self.latency_ms else None,
                "backend": self.backend,
                "confidence_note": "RF vote fraction, not a calibrated probability"}


class PortableForest:
    """Portable RF scorer; fixed NumPy weight arrays avoid sklearn pickle version issues."""
    def __init__(self, archive):
        self.classes_ = [str(v) for v in archive["classes"].tolist()]
        self.features = archive["features"]
        self.thresholds = archive["thresholds"]
        self.left = archive["left"]
        self.right = archive["right"]
        self.values = archive["values"]

    def predict_proba(self, inputs):
        results=[]
        for sample in np.asarray(inputs):
            probs=np.zeros(len(self.classes_), dtype=np.float64)
            for i in range(self.features.shape[0]):
                node=0
                while self.left[i,node]>=0:
                    if sample[self.features[i,node]]<=self.thresholds[i,node]:
                        node=int(self.left[i,node])
                    else:
                        node=int(self.right[i,node])
                votes=self.values[i,node,:]
                total=float(votes.sum())
                if total>0:probs+=votes/total
            probs/=self.features.shape[0]
            results.append(probs)
        return np.asarray(results)


class GestureRecognizer:
    """A trained, portable classifier; load the file once per program invocation."""

    def __init__(self, weights=DEFAULT_WEIGHTS, stream=False):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"Missing trained weights: {self.weights}. Run python src/train.py --backend contour")
        with np.load(self.weights, allow_pickle=False) as archive:
            self.schema=str(archive["schema"].item())
            if self.schema != "blurry-vision-npz-v1":
                raise ValueError("Unsupported model weight schema")
            self.backend = str(archive["backend"].item())
            self.estimator = PortableForest(archive)
        self.extractor = FeatureExtractor(self.backend, stream=stream)

    def close(self):
        self.extractor.close()

    def predict(self, image_bgr):
        start = time.perf_counter()
        feature, box, points, blur = self.extractor.extract(image_bgr)
        if feature is None:
            return Prediction("No hand detected", 0.0, None, None, blur,
                              (time.perf_counter() - start) * 1000, self.backend)
        probabilities = self.estimator.predict_proba(feature.reshape(1, -1))[0]
        i = int(np.argmax(probabilities))
        return Prediction(str(self.estimator.classes_[i]), float(probabilities[i]),
                          box, points, blur, (time.perf_counter() - start)*1000, self.backend)

def draw_result(image_bgr, prediction):
    """Render actual detected ROI/landmarks, not a made-up fixed rectangle."""
    h, w = image_bgr.shape[:2]
    base = cv2.resize(image_bgr, (max(224, w), max(224, h))) if min(h, w) < 224 else image_bgr.copy()
    h, w = base.shape[:2]
    if prediction.box is not None:
        x1, y1, x2, y2 = prediction.box
        xs, ys = w/224, h/224
        cv2.rectangle(base, (int(x1*xs), int(y1*ys)), (int(x2*xs), int(y2*ys)), (78, 224, 113), 2)
    if prediction.points:
        # MediaPipe's canonical 21-landmark hand skeleton. Drawing the links as
        # well as the points makes finger detection auditable to the user.
        connections = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                       (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                       (15,16),(13,17),(17,18),(18,19),(19,20),(0,17))
        for a, b in connections:
            ax, ay = prediction.points[a]
            bx, by = prediction.points[b]
            cv2.line(base, (int(ax*w/224), int(ay*h/224)),
                     (int(bx*w/224), int(by*h/224)), (191, 125, 255), 2, cv2.LINE_AA)
        for x, y in prediction.points:
            cv2.circle(base, (int(x*w/224), int(y*h/224)), 4, (0, 218, 255), -1)
    cv2.rectangle(base, (0, 0), (w, 65), (32, 26, 48), thickness=-1)
    cv2.putText(base, f"{prediction.label} | {prediction.confidence:.0%}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(base, f"{prediction.backend} | {prediction.latency_ms:.1f} ms | {1000/max(prediction.latency_ms, 1):.1f} FPS",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (194, 201, 208), 1, cv2.LINE_AA)
    return base


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
