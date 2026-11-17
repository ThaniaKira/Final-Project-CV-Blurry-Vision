"""Legacy API: genuine MediaPipe-only landmark classifier (NO fake fallback).

For reproducible deployed inference, use src/train.py and src/gesture_runtime.py.
The earlier version silently fabricated 21 contour 'landmarks' when MediaPipe
was missing. That behavior is intentionally removed to protect metric integrity.
"""
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.gesture_runtime import FeatureExtractor


class MediaPipeGestureClassifier:
    def __init__(self):
        self.extractor=FeatureExtractor("mediapipe")  # raises if unavailable
        self.clf=RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=1)
        self.is_trained=False

    def fit(self, X_images, y_labels):
        vectors=[];labels=[]
        for image,label in zip(X_images,y_labels):
            features,box,points,blur=self.extractor.extract(image)
            if features is not None:
                vectors.append(features);labels.append(label)
        if len(set(labels))<len(set(y_labels)):
            raise RuntimeError("MediaPipe missed gestures during training. Use real hand photographs; no fake features are generated")
        self.clf.fit(np.array(vectors),labels)
        self.is_trained=True

    def predict_single(self,image):
        if not self.is_trained:raise RuntimeError("Call fit() first")
        features,box,points,_=self.extractor.extract(image)
        if features is None:return "No hand detected",0.0,None
        probs=self.clf.predict_proba(features.reshape(1,-1))[0]
        index=int(np.argmax(probs))
        return str(self.clf.classes_[index]),float(probs[index]),points

    def predict_batch(self,images):
        return [self.predict_single(image)[0] for image in images]
