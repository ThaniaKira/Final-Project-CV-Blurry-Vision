# Real-Time Educational Hand Gesture Recognition for Interactive Early Childhood Learning

**Project Team**: Blurry Vision  
**Target Milestone**: Checkpoint 2 — 50% Progress Check (`cp2-50`)  
**Target Submission**: October 2026  
**Repository Name**: `Final_Project_CV_Blurry_Vision_Phase_2`  

---

## 📌 Executive Overview & Objectives

This repository contains the computer vision system for real-time educational hand gesture recognition designed for touchless interaction in early childhood learning environments. Operating via commodity webcams on CPU-only hardware, the system classifies 6 target gesture classes:

- **Open_Hand** (Open Palm / 5)
- **Fist** (0 / Closed)
- **Peace** (2 / V-sign)
- **Thumbs_Up** (Affirmative signal)
- **Pointing** (1 / Index finger)
- **OK** (Circle gesture)

### Measurable Technical Targets (From Proposal):
1. **Macro-F1 Target**: $\ge 0.85$ across all 6 gesture classes on the held-out test dataset.
2. **Real-Time Speed**: $\ge 15$ FPS on CPU-only standard laptops.
3. **Minimum Viable Baseline**: Classical YCbCr/HSV skin segmentation + contour/spatial feature baseline achieving macro-F1 $\ge 0.70$ at $\ge 20$ FPS.
4. **Robustness**: Classification accuracy $\ge 80\%$ under varied lighting conditions.

---

## 🚀 How to Run (Fresh Clone Quickstart)

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/ThaniaKira/Final-Project-CV-Blurry-Vision.git
cd Final-Project-CV-Blurry-Vision

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Step 2: Run Single-Command Working Prototype CLI
To run inference on any sample image and output an annotated prediction screenshot:
```bash
python src/predict.py --input data/samples/sample_01_Open_Hand.png --output outputs/result_sample_01.jpg
```

### Step 3: Run Baseline Validation Evaluator
To train the classical baseline model and print accuracy & Macro-F1 on the held-out validation set:
```bash
python src/baseline.py
```

### Step 4: Batch Generate All Output Screenshots
To process all 20 sample gesture images and update the `outputs/` directory:
```bash
python -m src.utils
```

---

## 📁 Repository Directory Layout

```text
Final_Project_CV_Blurry_Vision_Phase_2/
├── README.md                          # Quickstart guide & setup instructions
├── requirements.txt                    # Pinned python environment dependencies
├── .gitignore                          # Excludes temporary and environment files
├── data/                               # Dataset directory
│   ├── hagrid_sample_annotations.json # Metadata and bounding box annotations
│   ├── contact_sheet.png              # 20-sample grid contact sheet
│   ├── train/                         # Training split (70% by subject)
│   ├── val/                           # Validation split (15% by subject)
│   ├── test/                          # Locked test split (15% by subject)
│   └── samples/                       # 20 raw sample gesture PNG images
├── src/                                # Modular pipeline source code
│   ├── __init__.py
│   ├── preprocess.py                  # Shared Gaussian blur, YCbCr skin mask & Laplacian blur score
│   ├── data_loader.py                 # Dataset loader & 70/15/15 subject splitter
│   ├── baseline.py                    # Classical baseline model trainer & validation evaluator
│   ├── predict.py                     # Single-command CLI prototype predictor
│   └── utils.py                       # Contact sheet generator & batch screenshot utility
├── notebooks/                          # Exploration & feasibility spike notebooks
│   └── 01_feasibility_spike.ipynb     # Milestone 1 feasibility spike notebook
├── outputs/                            # Pre-populated with 20 annotated result screenshots
│   ├── result_sample_01_Open_Hand.jpg
│   ├── result_sample_02_Fist.jpg
│   └── ...
└── docs/                               # Milestone documentation & deliverables
    ├── cv-project-proposal.pdf        # Project proposal document
    ├── cv-project-proposal.docx       # Editable proposal document
    ├── data_card.md                   # One-page Data Card (Sources, splits, license)
    ├── progress_report.md             # Milestone 2 Progress Report (50% status & metrics)
    └── screenshot_instructions.md     # Guide for capturing output screenshots
```

---

## 🏷️ Milestone & Git Tag Reference
- **Milestone 1**: `cp1-proposal` (Proposal & Feasibility Spike)
- **Milestone 2**: `cp2-50` (50% Progress Check & Baseline Prototype)
