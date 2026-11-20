"""Real-time webcam; uses real hand detector by default, Q/Esc to exit.

Use --mode kaggle after genuine HaGRID training; --mode legacy for original
illustrated contour model. No accuracy is inferred from camera FPS.
"""
import os as _os,sys as _sys
_sys.path.insert(0,_os.path.abspath(_os.path.join(_os.path.dirname(__file__),"..")))
import argparse
from collections import deque,Counter
from pathlib import Path
import cv2
from src.gesture_runtime import GestureRecognizer,draw_result
from src.geometry_rules import GeometryRecognizer


def stream(webcam=0,mode="rules",smooth=5):
    if mode=="kaggle":
        weights=Path("models/gesture_hagrid_mediapipe.npz")
        if not weights.exists():
            raise FileNotFoundError("Kaggle weights missing; follow docs/KAGGLE_TRAINING.md")
        rec=GestureRecognizer(weights,stream=True)
    elif mode=="legacy":
        rec=GestureRecognizer("models/gesture_contour.npz",stream=True)
    else:
        rec=GeometryRecognizer(stream=True)
    cap=cv2.VideoCapture(webcam)
    if not cap.isOpened():
        rec.close(); raise RuntimeError(f"Cannot open camera {webcam}; check Windows camera permissions")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    recent=deque(maxlen=max(1,smooth))
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            pred=rec.predict(frame)
            if pred.box is not None and pred.label not in ("Uncertain gesture","No hand detected"):
                recent.append(pred.label)
                if len(recent)>=3: pred.label=Counter(recent).most_common(1)[0][0]
            else: recent.clear()
            cv2.imshow(f"Blurry Vision | {mode} | press Q to close",draw_result(frame,pred))
            if cv2.waitKey(1)&0xFF in (ord('q'),27):break
    finally:
        cap.release();rec.close();cv2.destroyAllWindows()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--webcam",type=int,default=0)
    parser.add_argument("--mode",choices=["rules","kaggle","legacy"],default="rules")
    parser.add_argument("--smooth",type=int,default=5)
    args=parser.parse_args()
    stream(args.webcam,args.mode,args.smooth)
