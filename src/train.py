"""Train a reproducible classifier from only data/train/ (never test/)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import GestureDataLoader
from src.gesture_runtime import FeatureExtractor


def train(backend="contour", output=None, root="data"):
    if output is None:
        output = "models/gesture_contour.npz" if backend == "contour" else "models/gesture_mediapipe.npz"
    images, labels, paths = GestureDataLoader(root).load_split("train")
    if not images:
        raise RuntimeError("data/train must contain labeled images; no fallback to test/ or samples/")
    extractor = FeatureExtractor(backend)
    features, y, valid_paths = [], [], []
    try:
        for image, label, path in zip(images, labels, paths):
            vector, _, _, _ = extractor.extract(image)
            if vector is not None:
                features.append(vector)
                y.append(label)
                valid_paths.append(path)
    finally:
        extractor.close()
    if len(set(y)) < len(set(labels)):
        missing = sorted(set(labels) - set(y))
        raise RuntimeError(f"{backend} could not extract hands for class(es): {missing}. "
                           "For MediaPipe, use real hand photographs, not the illustrative samples.")
    if len(features) < 6:
        raise RuntimeError("Too few valid training samples")
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
    clf.fit(np.asarray(features), y)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    # Export a version-neutral NumPy archive of the trained tree arrays.
    # A sklearn 1.8 pickled RF may fail under a 1.5 classroom environment.
    trees=[est.tree_ for est in clf.estimators_]
    count=len(trees); nodes=max(t.node_count for t in trees); nclass=len(clf.classes_)
    feat=np.zeros((count,nodes),dtype=np.int32)
    thresholds=np.zeros((count,nodes),dtype=np.float64)
    left=np.full((count,nodes),-1,dtype=np.int32)
    right=np.full((count,nodes),-1,dtype=np.int32)
    values=np.zeros((count,nodes,nclass),dtype=np.float64)
    for i,t in enumerate(trees):
        n=t.node_count
        feat[i,:n]=t.feature
        thresholds[i,:n]=t.threshold
        left[i,:n]=t.children_left
        right[i,:n]=t.children_right
        values[i,:n,:]=t.value[:,0,:]
    np.savez_compressed(output,schema="blurry-vision-npz-v1",backend=backend,
                        classes=np.asarray(clf.classes_,dtype="U32"),
                        features=feat,thresholds=thresholds,left=left,right=right,values=values,
                        training_count=len(y),training_paths=np.asarray(valid_paths,dtype="U255"),seed=42)
    print(f"Saved {output}: backend={backend}; training_samples={len(y)}; classes={list(clf.classes_)}")
    return output


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend", choices=["contour", "mediapipe"], default="contour")
    p.add_argument("--output", default=None)
    p.add_argument("--data", default="data")
    a=p.parse_args()
    train(a.backend, a.output, a.data)
