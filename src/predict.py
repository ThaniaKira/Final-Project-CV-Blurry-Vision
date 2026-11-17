"""Single-image backwards-compatible CLI."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import cv2
from src.pipeline import run

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="outputs/result_prediction.jpg")
    p.add_argument("--weights",default="models/gesture_contour.npz")
    a=p.parse_args()
    image=cv2.imread(a.input)
    if image is None:raise FileNotFoundError(f"Cannot read {a.input}")
    result,detail=run(image,a.weights)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(a.output,result)
    print(detail)
    print("Saved",a.output)
