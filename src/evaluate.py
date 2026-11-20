"""Deterministic held-out evaluation, per-class metrics and warm-start CPU latency.

Example: python src/evaluate.py --split test --weights models/gesture_contour.npz
Produces outputs/eval/metrics.json, confusion_matrix.png and predictions.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from src.data_loader import GestureDataLoader
from src.baseline import ClassicalBaselineModel
from src.gesture_runtime import GestureRecognizer, CLASSES, sha256_file, draw_result


def measure(model, images, predictions):
    # Warm-up is not included in the measurement; includes full preprocessing.
    for image in images[:min(3, len(images))]:
        model.predict(image)
    times = []
    for image in images:
        before = time.perf_counter()
        model.predict(image)
        times.append((time.perf_counter()-before) * 1000)
    return float(np.mean(times)) if times else None


def evaluate(split="test", weights="models/gesture_contour.npz", data="data", out="outputs/eval"):
    if split != "test" and split != "val":
        raise ValueError("Only the frozen test or validation split is supported")
    images, labels, paths = GestureDataLoader(data).load_split(split)
    real_dataset = (Path(data)/"data_card.json").exists()
    if real_dataset:
        # Revalidate the disjoint-subject condition instead of trusting a report.
        import csv as _csv
        manifest_path=Path(data)/"manifest.csv"
        if not manifest_path.exists():
            raise RuntimeError("Real HaGRID evaluation requires data/manifest.csv for subject-disjoint audit")
        with manifest_path.open(newline="",encoding="utf-8") as f:
            rows=list(_csv.DictReader(f))
        groups={sp:{row["user_id"] for row in rows if row["split"]==sp} for sp in ("train","val","test")}
        if (groups["train"] & groups["val"]) or (groups["train"] & groups["test"]) or (groups["val"] & groups["test"]):
            raise RuntimeError("Subject leakage found between HaGRID partitions; results invalid")
    train_images, train_labels, _ = GestureDataLoader(data).load_split("train")
    if not images or not train_images:
        raise RuntimeError("Missing dataset split; evaluation never substitutes training files")
    Path(out).mkdir(parents=True, exist_ok=True)
    model = GestureRecognizer(weights)
    if model.backend=="mediapipe" and not real_dataset:
        model.close()
        raise RuntimeError("MediaPipe Kaggle model must be evaluated on real subject-disjoint HaGRID data. "
                           "Specify --data data/hagrid_real")
    if model.backend=="contour" and real_dataset:
        print("NOTE: Evaluating legacy contour backend on real HaGRID data (diagnostic comparison only).")
    try:
        preds = [model.predict(im) for im in images]
        y_pred = [p.label for p in preds]
        latency = measure(model, images, preds)
        main_report = classification_report(labels, y_pred, labels=list(CLASSES), output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, y_pred, labels=list(CLASSES))
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=list(CLASSES)).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
        ax.set_title(f"{model.backend} model — {split} set (n={len(labels)})")
        fig.tight_layout()
        fig.savefig(Path(out)/"confusion_matrix.png", dpi=140)
        plt.close(fig)
        with open(Path(out)/"predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["filepath","true_label","predicted_label","confidence","correct"])
            for path, label, pred in zip(paths, labels, preds):
                writer.writerow([path,label,pred.label,f"{pred.confidence:.5f}",str(pred.label==label)])
        # Real evidence gallery, with both successes and failures (avoid only
        # displaying the model's favorite easy examples). Source is TEST only.
        gallery_dir=Path(out)/"gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_rows=[]
        per_kind={}
        for image, path, actual, prediction in zip(images,paths,labels,preds):
            kind = "correct" if actual==prediction.label else "error"
            key=(actual,kind)
            cap=1 if kind=="correct" else 3
            if per_kind.get(key,0)>=cap:
                continue
            per_kind[key]=per_kind.get(key,0)+1
            filename=(f"{kind}_{actual}_{Path(path).stem}".replace(" ","_")+".jpg")
            target=gallery_dir/filename
            cv2.imwrite(str(target), draw_result(image,prediction))
            gallery_rows.append({"file":str(target),"ground_truth":actual,
                                 "predicted":prediction.label,"correct":actual==prediction.label})
        baseline = ClassicalBaselineModel(n_neighbors=1)
        baseline.fit(train_images, train_labels)
        b_preds = baseline.predict(images)
        # Fair latency measure: same test images; warm the fitted baseline first.
        baseline.predict(images[:min(3,len(images))])
        b_start = time.perf_counter()
        baseline.predict(images)
        b_latency=(time.perf_counter()-b_start)/len(images)*1000
        result = {
            "split":split,"n_test":len(images),"class_distribution":{c:int(labels.count(c)) for c in CLASSES},
            "backend":model.backend,"weights_path":str(weights),"weights_sha256":sha256_file(weights),
            "data_path":str(data),
            "test_set_note":("HaGRID cropped real hand photos, subject-disjoint by user_id; "
                             "webcam/generalization must be separately tested" if real_dataset
                             else "Only six illustrated test images in the original ZIP; not webcam evidence"),
            "gallery_examples":gallery_rows,
            "no_hand_detected":int(sum(p.label=="No hand detected" for p in preds)),
            "hand_detection_rate":float(sum(p.box is not None for p in preds)/len(preds)),
            "missing_class_handling":"No hand detected counts as incorrect for ground-truth gesture",
            "main":{"accuracy":float(accuracy_score(labels,y_pred)),
                    "macro_f1":float(f1_score(labels,y_pred,labels=list(CLASSES),average="macro",zero_division=0)),
                    "latency_ms":latency,"fps":1000/latency if latency else None,
                    "per_class":{c:{k:float(main_report[c][k]) if k!='support' else int(main_report[c][k]) for k in ('precision','recall','f1-score','support')} for c in CLASSES}},
            "baseline":{"accuracy":float(accuracy_score(labels,b_preds)),
                        "macro_f1":float(f1_score(labels,b_preds,labels=list(CLASSES),average="macro",zero_division=0)),
                        "latency_ms":b_latency,"fps":1000/b_latency if b_latency>0 else None},
            "hardware":{"platform":platform.platform(),"python":platform.python_version(),
                        "opencv":cv2.__version__,"numpy":np.__version__},
            "comparison_to_cp3":("Different dataset from CP3 illustrations: do NOT compare as a percent improvement. "
                                  "Report the original and HaGRID metrics separately." if real_dataset else
                                  "Original six-illustration results do not establish real webcam accuracy."),
        }
        out_path=Path(out)/"metrics.json"
        out_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print("\nBLURRY VISION | REPRODUCIBLE EVALUATION")
        print(f"Split: {split}, n={len(images)}, backend: {model.backend}, weights SHA256: {result['weights_sha256'][:16]}...")
        print(f"Baseline: accuracy={result['baseline']['accuracy']:.4f}, macro-F1={result['baseline']['macro_f1']:.4f}, latency={b_latency:.2f} ms/image")
        print(f"Main:     accuracy={result['main']['accuracy']:.4f}, macro-F1={result['main']['macro_f1']:.4f}, latency={latency:.2f} ms/image, FPS={result['main']['fps']:.1f}")
        print(classification_report(labels,y_pred, labels=list(CLASSES), zero_division=0))
        print(f"Saved {out_path}, {Path(out)/'confusion_matrix.png'}, predictions.csv")
        if real_dataset:
            print(f"REAL HAND DATA: {len(images)} images, genuine detected hands: {result['hand_detection_rate']:.1%}; "
                  "source-subject separation verified. Webcam testing still required.")
        else:
            print("CAUTION: Six illustrated images are not enough for claims about webcam performance.")
        return result
    finally:
        model.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test","val"])
    p.add_argument("--weights", default="models/gesture_contour.npz")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="outputs/eval")
    args=p.parse_args()
    evaluate(args.split,args.weights,args.data,args.output)
