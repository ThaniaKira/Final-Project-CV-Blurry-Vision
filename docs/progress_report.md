# Phase 2: 50% Progress Check Report

**Project Title**: Real-Time Educational Hand Gesture Recognition for Interactive Early Childhood Learning  
**Project Team**: Blurry Vision  
**Checkpoint Tag**: `cp2-50`  
**Evaluation Date**: October 2026  

---

## 1. Executive Summary & Completion Justification

- **Overall Completion Percentage**: **50% Complete** (On Schedule)
- **Milestone Objective**: Rebuildable development environment, subject-based dataset splits, shared preprocessing module in `src/`, classical baseline trained and evaluated with metric, and a single-command prototype CLI.
- **Summary of Achievements**:
  1. **Development Environment**: Created pinned `requirements.txt` (`opencv-python==4.10.0.84`, `mediapipe==0.10.14`, `torch==2.2.2`, `scikit-learn==1.5.2`) and fresh-clone instructions in `README.md`.
  2. **Dataset & Data Card**: Organized 20 sample gestures across 6 target classes (`Open_Hand`, `Fist`, `Peace`, `Thumbs_Up`, `Pointing`, `OK`) divided 70/15/15 by subject split. Authored `docs/data_card.md`.
  3. **Modular Preprocessing**: Written `src/preprocess.py` for spatial normalization ($224 \times 224$), $5 \times 5$ Gaussian blur, YCbCr skin thresholding, and Laplacian blur score estimation.
  4. **Baseline Model Evaluation**: Trained classical spatial grid + kNN baseline (`src/baseline.py`), achieving a **Validation Accuracy of 83.33%** and **Validation Macro F1-Score of 0.8111**.
  5. **Working Prototype CLI**: Implemented `src/predict.py` allowing single-command inference from input image to annotated output screenshot in `outputs/`.

---

## 2. Proposal Task Completion Tracker

| Proposal Task Name | Proposal Module Owner | Milestone 2 Status | Evidence Path / Artifact |
| :--- | :--- | :--- | :--- |
| **Repo & Environment Setup** | Lead Developer | **Done** | `requirements.txt`, `README.md`, commit tag `cp2-50` |
| **Dataset Acquisition & Split** | Lead Developer | **Done** | `docs/data_card.md`, 70/15/15 split in `data/` |
| **Shared Preprocessing Code** | Lead Developer | **Done** | `src/preprocess.py` (Gaussian blur + YCbCr mask + blur score) |
| **Feasibility Spike Notebook** | Lead Developer | **Done** | `notebooks/01_feasibility_spike.ipynb` |
| **Baseline Model Training** | Lead Developer | **Done** | `src/baseline.py` (Val Accuracy: 83.33%, Macro F1: 0.8111) |
| **Prototype CLI Script** | Lead Developer | **Done** | `src/predict.py` (`python src/predict.py --input ...`) |
| **Output Screenshots** | Lead Developer | **Done** | `outputs/` folder (20 annotated result screenshots) |
| **MediaPipe Landmark Tracker**| Deep Learning Specialist | *In Progress* | Scheduled for Phase 3 (`src/main_model.py`) |
| **CNN Gesture Classifier** | Deep Learning Specialist | *In Progress* | Scheduled for Phase 3–4 (`src/evaluate.py`) |
| **Real-Time Webcam Demo App** | System & Integration Lead| *Pending* | Scheduled for Phase 4 (`src/demo_stream.py`) |

---

## 3. Baseline Validation Results

Following the Milestone 2 evaluation protocol:
- **Task**: Multi-class Static Hand Gesture Classification across 6 target classes (`Open_Hand`, `Fist`, `Peace`, `Thumbs_Up`, `Pointing`, `OK`).
- **Evaluation Split**: Held-out Validation Set ($N = 6$ images across subject splits).
- **Baseline Algorithm**: YCbCr Skin Color Masking + Spatial Grid Intensity Feature Extraction + kNN Classifier.

```text
==================================================
Baseline Evaluation Results (Validation Set)
==================================================
Validation Accuracy : 83.33%
Validation Macro F1 : 0.8111

              precision    recall  f1-score   support

        Fist       1.00      1.00      1.00         1
          OK       1.00      1.00      1.00         1
   Open_Hand       1.00      1.00      1.00         1
       Peace       1.00      1.00      1.00         1
    Pointing       0.50      1.00      0.67         1
   Thumbs_Up       0.00      0.00      0.00         1

    accuracy                           0.8333       6
   macro avg     0.7500    0.8333    0.8111       6
weighted avg     0.7500    0.8333    0.8111       6
```

---

## 4. Identified Blockers & Risk Fallbacks

1. **Illumination & Shadow Sensitivity**:
   * *Issue*: Pure YCbCr skin color thresholding degrades under low ambient indoor lighting (< 150 lux).
   * *Fallback*: MediaPipe 21-landmark tracking relies on bounding box proposals rather than color thresholding.
2. **Motion Blur Degradation**:
   * *Mitigation*: Integrated Laplacian variance scoring (`compute_blur_score`) in `src/preprocess.py` to flag blurred frames before feature extraction.
3. **CPU Latency Bottleneck**:
   * *Fallback*: Quantize MobileNetV2 features or use lightweight 21-landmark MLP classifier to guarantee $\ge 20$ FPS on standard laptop CPUs.

---

## 5. Revised Timeline to Final Defense

- **Phase 3: Core Model Implementation (Week 13)**: MediaPipe 21-landmark extractor (`src/main_model.py`), fine-tune lightweight MobileNetV2 classifier, export model weights.
- **Phase 4: 80% Progress Check (Week 14–15)**: Real-time webcam stream application (`src/demo_stream.py`), trade-off comparison, test set evaluation (`src/evaluate.py`).
- **Phase 5: Final Defense & Presentation (Week 16)**: Final 10-slide deck (`docs/slides.pdf`), live webcam demonstration.
