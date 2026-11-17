from pathlib import Path
import numpy as np
from src.gesture_runtime import FeatureExtractor, GestureRecognizer


def test_contour_blank_returns_none():
    x=np.full((224,224,3),255,dtype=np.uint8)
    feature,box,points,blur=FeatureExtractor("contour").extract(x)
    assert feature is None and box is None


def test_weights_metadata():
    path=Path("models/gesture_contour.npz")
    if path.exists():
        model=GestureRecognizer(path)
        assert model.backend=="contour"
        model.close()
