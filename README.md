# Computer Vision: Real-Time Educational Hand Gesture Recognition System

**Project Title**: Real-Time Educational Hand Gesture Recognition for Interactive Early Childhood Learning
**Team Name**: Blurry Vision
**Target Checkpoint**: Checkpoint 3 (Milestone 3 - Core Model Implementation & Testing)
**Git Release Tag Target**: `cp3-core`
**Repository URL**: `https://github.com/ThaniaKira/Final-Project-CV-Blurry-Vision.git`

---

## 📌 Project Overview
This project delivers a real-time, touchless hand gesture recognition system designed for interactive early childhood educational tools. Utilizing standard commercial webcams without depth/infrared sensors, the system identifies **6 target static hand gestures**:
1. **Open_Hand** (Palm open / counting 5)
2. **Fist** (Hand closed / counting 0)
3. **Peace** (V-sign / counting 2)
4. **Thumbs_Up** (Affirmative / like signal)
5. **Pointing** (Index extended / selection signal)
6. **OK** (OK ring gesture / confirmation signal)

---

## ⚙️ Development Environment Setup

### 1. Fresh Clone & Virtual Environment Setup
```bash
# Clone repository
git clone https://github.com/ThaniaKira/Final-Project-CV-Blurry-Vision.git
cd Final-Project-CV-Blurry-Vision

# Create and activate Python virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\activate

# Activate on Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Pipeline (Milestone 3 Single Command)

### 1. Run Pipeline on Image or Folder
Process an input image or folder of gesture samples with a single command:
```bash
# Run end-to-end pipeline on sample folder
python src/run.py --input data/samples --output outputs/run_results

# Run on a single image file
python src/run.py --input data/samples/sample_01_Open_Hand.png --output outputs/
```

### 2. Single Image CLI Prediction
```bash
python -m src.predict --input "data/samples/sample_01_Open_Hand.png" --output "outputs/result_sample_01.jpg"
```

### 3. Run Benchmark Evaluation & Generate Metrics
Evaluate the Classical Baseline vs MediaPipe Core Landmark Classifier on the held-out test split:
```bash
python src/evaluate.py
```
*Outputs accuracy, macro-F1, per-class precision/recall, inference FPS, and saves the **Confusion Matrix Display** plot to `outputs/confusion_matrix.png`.*

### 4. Batch Process All Samples & Rebuild Contact Sheet
```bash
python -m src.utils
```

---

## 📁 Repository Directory Layout

```text
Final_Project_CV_Blurry_Vision_Phase_3_CP3/
├── README.md                          # Quickstart guide & fresh clone instructions
├── requirements.txt                    # Pinned python environment dependencies
├── .gitignore                          # Excludes temporary & virtualenv files
├── data/                               # Dataset directory
│   ├── hagrid_sample_annotations.json # Metadata and bounding box annotations
│   ├── contact_sheet.png              # 20-sample grid contact sheet
│   ├── train/                         # Training split (70% by subject)
│   ├── val/                           # Validation split (15% by subject)
│   ├── test/                          # Locked test split (15% by subject)
│   └── samples/                       # 20 attached sample gesture PNG images
├── src/                                # Modular pipeline source code
│   ├── __init__.py
│   ├── preprocess.py                  # Gaussian blur, YCbCr skin mask, CLAHE & blur score
│   ├── data_loader.py                 # Subject-based 70/15/15 dataset loader
│   ├── baseline.py                    # Classical baseline model evaluator
│   ├── main_model.py                  # MediaPipe 21 3D landmark extractor + classifier
│   ├── run.py                         # Single-command end-to-end pipeline runner
│   ├── predict.py                     # Single-image CLI predictor
│   ├── evaluate.py                    # Benchmark evaluation & confusion matrix builder
│   └── utils.py                       # Visualizations & contact sheet generator
├── notebooks/                          # Feasibility & model exploration
│   ├── 01_feasibility_spike.ipynb     # Milestone 1 feasibility spike
│   └── 02_core_model_experiments.ipynb# Milestone 3 MediaPipe landmark experiments
├── outputs/                            # Result screenshots & confusion matrix plot
│   ├── confusion_matrix.png           # Evaluated confusion matrix
│   └── ...                            # Pre-generated annotated screenshots
└── docs/                               # Milestone documentation
    ├── data_card.md                   # One-page Data Card
    ├── milestone3_report.md           # Milestone 3 Progress Report & Error Analysis
    └── cv-project-proposal.docx       # Project Proposal document
```

---

## 🏷️ Milestone & Git Tag Reference
- **Milestone 1**: `cp1-proposal` (Proposal & Feasibility Spike)
- **Milestone 2**: `cp2-50` (50% Progress Check & Baseline Prototype)
- **Milestone 3**: `cp3-core` (Core Model Implementation & Testing)