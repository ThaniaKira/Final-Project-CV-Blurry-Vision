"""Continuous local webcam demo with true frame timings. Press Q or Esc to exit."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
from collections import deque,Counter
import cv2
from src.gesture_runtime import GestureRecognizer, draw_result


def stream(webcam=0,weights="models/gesture_contour.npz",smooth=5):
    rec=GestureRecognizer(weights,stream=True)
    cap=cv2.VideoCapture(webcam)
    if not cap.isOpened():
        rec.close()
        raise RuntimeError(f"Cannot open webcam {webcam}; check OS permission, device ID and camera use")
    recent=deque(maxlen=max(1,smooth))
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            pred=rec.predict(frame)
            if pred.box is not None:
                recent.append(pred.label)
                pred.label=Counter(recent).most_common(1)[0][0] if len(recent)>=3 else pred.label
            else:
                recent.clear()
            cv2.imshow("Blurry Vision | Q/Esc to close",draw_result(frame,pred))
            if cv2.waitKey(1)&0xFF in (ord('q'),27):break
    finally:
        cap.release()
        rec.close()
        cv2.destroyAllWindows()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--webcam",type=int,default=0)
    p.add_argument("--weights",default="models/gesture_contour.npz")
    p.add_argument("--smooth",type=int,default=5)
    a=p.parse_args()
    stream(a.webcam,a.weights,a.smooth)
