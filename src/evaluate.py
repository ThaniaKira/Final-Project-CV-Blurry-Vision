import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from src.data_loader import GestureDataLoader
from src.baseline import ClassicalBaselineModel
from src.main_model import MediaPipeGestureClassifier

def evaluate_all():
    """Runs rigorous evaluation comparing Classical Baseline vs MediaPipe Core Model."""
    loader = GestureDataLoader()
    train_imgs, train_lbls, _ = loader.load_split("train")
    test_imgs, test_lbls, _ = loader.load_split("test")
    
    if len(test_imgs) == 0:
        train_imgs, train_lbls, _ = loader.load_split("samples")
        test_imgs, test_lbls = train_imgs[10:], train_lbls[10:]
        train_imgs, train_lbls = train_imgs[:10], train_lbls[:10]
        
    print("==================================================")
    print("MILESTONE 3: CORE MODEL EVALUATION & BENCHMARKING")
    print("==================================================")
    print(f"Held-out Test Dataset Size: {len(test_imgs)} samples across 6 classes")
    
    # 1. Classical Baseline Model
    baseline = ClassicalBaselineModel()
    baseline.fit(train_imgs, train_lbls)
    
    t0 = time.time()
    b_preds = baseline.predict(test_imgs)
    b_latency = (time.time() - t0) / len(test_imgs) * 1000.0
    b_fps = 1000.0 / b_latency
    b_acc = accuracy_score(test_lbls, b_preds)
    b_f1 = f1_score(test_lbls, b_preds, average="macro", zero_division=0)
    
    # 2. Main Method (MediaPipe Landmark Classifier)
    main_model = MediaPipeGestureClassifier()
    main_model.fit(train_imgs, train_lbls)
    
    t1 = time.time()
    m_preds = main_model.predict_batch(test_imgs)
    m_latency = (time.time() - t1) / len(test_imgs) * 1000.0
    m_fps = 1000.0 / m_latency
    m_acc = accuracy_score(test_lbls, m_preds)
    m_f1 = f1_score(test_lbls, m_preds, average="macro", zero_division=0)
    
    print("--------------------------------------------------")
    print("MODEL PERFORMANCE COMPARISON (TEST SPLIT)")
    print("--------------------------------------------------")
    print(f"1. Classical Baseline (Skin YCbCr + kNN):")
    print(f"   - Accuracy    : {b_acc*100:.2f}%")
    print(f"   - Macro-F1    : {b_f1:.4f}")
    print(f"   - Latency     : {b_latency:.2f} ms/frame ({b_fps:.1f} FPS)")
    
    print(f"\n2. Main Method (MediaPipe 21 Landmarks + RF):")
    print(f"   - Accuracy    : {m_acc*100:.2f}%")
    print(f"   - Macro-F1    : {m_f1:.4f}")
    print(f"   - Latency     : {m_latency:.2f} ms/frame ({m_fps:.1f} FPS)")
    
    print("\n--------------------------------------------------")
    print("MAIN METHOD DETAILED CLASSIFICATION REPORT")
    print("--------------------------------------------------")
    report_str = classification_report(test_lbls, m_preds, zero_division=0, digits=3)
    print(report_str)
    
    # Plot and save confusion matrix
    classes = ["Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK"]
    cm = confusion_matrix(test_lbls, m_preds, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title("Core Model Confusion Matrix (MediaPipe Landmarks)", fontsize=12)
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix plot saved to: {cm_path}")
    
    return m_acc, m_f1, m_fps

if __name__ == "__main__":
    evaluate_all()
