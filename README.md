# Real-Time Educational Hand Gesture Recognition

An interactive computer vision system designed for early childhood learning classrooms. By replacing expensive touch/sensor hardware with standard webcams, this system recognizes hand gestures in real-time to drive educational software interfaces.

---

## Folder Organization

```text
cv-project/
├── README.md                   # Project overview, setup, and execution guide
├── requirements.txt            # Python dependencies (OpenCV, MediaPipe, KaggleHub, PyTorch)
├── .gitignore                  # Git exclusions for datasets, checkpoints, and cache
├── docs/                       # Project documentation
│   ├── cv-project-proposal.pdf # Formal 3-page Checkpoint 1 Proposal
│   └── cv-project-proposal.docx# Editable Word version of proposal
├── data/                       # Dataset handling
│   ├── download_hagrid.py      # KaggleHub script to fetch HaGRID gesture dataset
│   └── samples/                # 20 labeled sample gesture images
├── notebooks/                  # Colab & Jupyter experiment notebooks
│   └── 01_feasibility_spike.ipynb # Checkpoint 1 Feasibility Spike (HaGRID + Vision OP)
├── src/                        # Core source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Data pipeline & 70/15/15 Train/Val/Test splitting
│   ├── baseline.py             # Classical CV baseline (Skin segmentation + SVM)
│   └── utils.py                # Helper functions, metrics, and visualization
└── outputs/                    # Generated visuals and figures
    ├── contact_sheet.png       # 20-sample dataset contact sheet
    └── spike_screenshot.png    # Feasibility spike output screenshot
```

---

## Dataset Setup (HaGRID)

We use the HaGRID (Hand Gesture Recognition Image Dataset) hosted on Kaggle via kagglehub.

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Download Dataset
Run the automated downloader script:
```bash
python data/download_hagrid.py
```

---

## Execution Guide

### 1. Run the Feasibility Spike Notebook
Open `notebooks/01_feasibility_spike.ipynb` in Google Colab or locally.

### 2. Run Data Loader Split
```bash
python src/data_loader.py
```

### 3. Run Classical Baseline Model
```bash
python src/baseline.py
```

---

