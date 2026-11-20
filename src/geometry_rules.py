"""Explainable real-hand, single-snapshot gesture *heuristic* fallback.

This does not constitute training on HaGRID; do not report a model accuracy
unless these rules were evaluated on held-out photographs.
"""
from __future__ import annotations
import numpy as np
from src.gesture_runtime import FeatureExtractor, Prediction
import time


def angle(a, b, c):
    u, v = a - b, c - b
    denom = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denom, -1., 1.))))


def classify_geometry(points):
    """Coordinates shape=(21,2), in pixel units (aspect-ratio corrected)."""
    p=np.asarray(points,dtype=float)
    wrist=p[0]; palm=max(float(np.linalg.norm(p[9]-wrist)), 1e-6)
    extended={}
    for name,mcp,pip,tip in (("index",5,6,8),("middle",9,10,12),("ring",13,14,16),("pinky",17,18,20)):
        extended[name]= bool(angle(p[mcp],p[pip],p[tip])>152 and np.linalg.norm(p[tip]-wrist)>np.linalg.norm(p[pip]-wrist)*1.065)
    thumb_straight=angle(p[1],p[2],p[4])>142
    thumb_out=thumb_straight and np.linalg.norm(p[4]-p[17])>np.linalg.norm(p[3]-p[17])*1.10
    pinch=float(np.linalg.norm(p[4]-p[8])/palm)<.43
    n=sum(extended.values())
    # Evaluate distinctive multi-finger poses before the more general rules.
    # The thumb is measured separately because its anatomy differs from the
    # other fingers in MediaPipe's landmark layout.
    if pinch and n == 0:
        # Korean finger-heart: thumb and curled index fingertip meet, while
        # the other fingers stay folded. It otherwise resembles Thumbs Up.
        label="Finger_Heart"
    elif thumb_out and extended["pinky"] and n == 1:
        label="Call_Me"
    elif thumb_out and extended["index"] and extended["pinky"] and n == 2:
        label="I_Love_You"
    elif extended["index"] and extended["pinky"] and n == 2 and not thumb_out:
        label="Rock"
    elif extended["index"] and extended["middle"] and extended["ring"] and n == 3:
        label="Three_Fingers"
    elif pinch and extended["middle"] and extended["ring"]:
        label="OK"
    elif n==4:
        label="Open_Hand"
    elif extended["index"] and extended["middle"] and not (extended["ring"] or extended["pinky"]):
        label="Peace"
    elif extended["index"] and not (extended["middle"] or extended["ring"] or extended["pinky"]) and not pinch:
        label="Pointing"
    elif n==0 and thumb_out:
        # In image coordinates y grows downward. A fully extended thumb below
        # the wrist is a thumbs-down/dislike pose, not a thumbs-up. This is
        # only a transparent fallback rule; the HaGRID model remains the
        # correct path for robust multi-gesture recognition.
        label="Dislike" if p[4, 1] > wrist[1] + .25 * palm else "Thumbs_Up"
    elif n==0 and not thumb_out:
        label="Fist"
    else:
        label="Uncertain gesture"
    return label, {"fingers_extended":extended,"thumb_extended":bool(thumb_out),"pinch":bool(pinch)}


class GeometryRecognizer:
    backend="mediapipe-rules"
    def __init__(self, stream=False):
        self.extractor=FeatureExtractor("mediapipe",stream=stream)

    def predict(self,image_bgr):
        start=time.perf_counter()
        _,box,points,blur=self.extractor.extract(image_bgr)
        if points is None:
            return Prediction("No hand detected",0.0,None,None,blur,(time.perf_counter()-start)*1000,self.backend)
        # FeatureExtractor uses 224-reference coordinates, preserving aspect ratio.
        h,w=image_bgr.shape[:2]
        label,_=classify_geometry([[p[0]*w/224,p[1]*h/224] for p in points])
        return Prediction(label,0.0,box,points,blur,(time.perf_counter()-start)*1000,self.backend)

    def close(self): self.extractor.close()
