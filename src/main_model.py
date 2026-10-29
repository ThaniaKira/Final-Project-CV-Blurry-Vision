import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.preprocess import preprocess_pipeline

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class DummyHandLandmarks:
    def __init__(self, points):
        self.landmark = [DummyLandmark(p[0], p[1], p[2] if len(p)>2 else 0.0) for p in points]

class HandLandmarkExtractor:
    """Extracts 21 3D normalized hand landmark coordinates (63-dim feature vector)."""
    def __init__(self, static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3):
        self.has_mp = HAS_MEDIAPIPE
        if self.has_mp:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence
            )

    def extract_landmarks(self, image):
        """Returns 63-dim normalized landmark array (x, y, z for 21 points) and landmark object."""
        if self.has_mp:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_vector = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    landmarks_vector.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                return np.array(landmarks_vector, dtype=np.float32), hand_landmarks

        # OpenCV Keypoint & Contour Landmark Extractor Fallback
        _, mask, _, _ = preprocess_pipeline(image, target_size=(224, 224))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = M["m10"]/M["m00"] if M["m00"] != 0 else 112.0
            cy = M["m01"]/M["m00"] if M["m00"] != 0 else 112.0
            
            # Extract 21 spatial contour/hull keypoints
            hull = cv2.convexHull(c, returnPoints=True)
            pts_source = hull if len(hull) >= 21 else c
            
            for i in range(21):
                pt = pts_source[i % len(pts_source)][0]
                norm_x = (pt[0] - cx) / 224.0
                norm_y = (pt[1] - cy) / 224.0
                points.append((norm_x, norm_y, 0.0))
        else:
            # Uniform 21-point dummy grid
            for i in range(21):
                points.append((i * 0.01, i * 0.01, 0.0))
                
        vec = []
        wrist_x, wrist_y = points[0][0], points[0][1]
        for p in points:
            vec.extend([p[0] - wrist_x, p[1] - wrist_y, p[2]])
            
        dummy_lm_obj = DummyHandLandmarks([(p[0]+0.5, p[1]+0.5, p[2]) for p in points])
        return np.array(vec, dtype=np.float32), dummy_lm_obj

class MediaPipeGestureClassifier:
    """Core Deep Learning / Landmark Classifier (MediaPipe 21 Landmarks + Ensemble Head)."""
    def __init__(self):
        self.extractor = HandLandmarkExtractor()
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classes = ["Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK"]
        self.is_trained = False

    def fit(self, X_images, y_labels):
        X_feats = []
        for img in X_images:
            feat, _ = self.extractor.extract_landmarks(img)
            X_feats.append(feat)
        self.clf.fit(X_feats, y_labels)
        self.is_trained = True

    def predict_single(self, image):
        feat, hand_landmarks = self.extractor.extract_landmarks(image)
        probs = self.clf.predict_proba([feat])[0]
        max_idx = np.argmax(probs)
        pred_label = self.clf.classes_[max_idx]
        confidence = float(probs[max_idx])
        return pred_label, confidence, hand_landmarks

    def predict_batch(self, X_images):
        return [self.predict_single(img)[0] for img in X_images]
