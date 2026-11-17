# Blurry Vision | Real-Time Educational Hand Gesture Recognition

**Checkpoints:** CP4 80% package, with final packaging guidance. **Scope:** six static gesture classes (Open_Hand, Fist, Peace, Thumbs_Up, Pointing, OK). The browser application accepts uploaded images or webcam *snapshots*. The desktop OpenCV application displays a continuous webcam feed. Outputs include a class, confidence (random-forest voting proportion), detected contour box or genuine MediaPipe landmarks, and processing time.

> **Verification warning:** The uploaded project contains only **20 illustrated gesture PNGs** (8 train, 6 val, 6 test), not a validated real-hand dataset. The included demo weights use **classical contour features + Random Forest**. They are *not* MediaPipe weights and are not validated on real hands. CP3's historical 100% MediaPipe claim was not independently reproducible from the archive. This package must not be described as classroom-ready or a proven 80% implementation without further verification.

## Quick start (recommended Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/app.py
```

Linux/macOS: `source .venv/bin/activate`. Open the local Gradio URL (usually `http://127.0.0.1:7860`). This is the **one command to launch the app after installation**. You can use an image file or webcam snapshot. The model file `models/gesture_contour.npz` is included.

**Local continuous webcam (press Q to close):**

```bash
python src/demo_stream.py --webcam 0
```

The included weights may misclassify real hands, especially if skin colors/backgrounds or conditions differ from the illustrations. Use genuine data before presenting it as a real-world system. MediaPipe 0.10.14 requires a compatible Python version (the included requirements are intended for 3.10–3.11).

## Reproduce reported numbers exactly on the included split

```bash
python src/evaluate.py --split test --weights models/gesture_contour.npz
```

The program prints accuracy, macro-F1, class metrics and **measured** latency, and writes `outputs/eval/metrics.json`, `outputs/eval/confusion_matrix.png`, and `outputs/eval/predictions.csv`. Accuracy is reproducible; FPS depends on the hardware, loads and runtime. To retrain the *classical contour model* without ever reading test data:

```bash
python src/train.py --backend contour --output models/gesture_contour.npz
```

To run all **12 original + edge-case** stress probes:

```bash
python src/test_cases.py
```

Sample images: `data/test/*/*.png`. Batch processing:

```bash
python src/run.py --input data/test --output outputs/new_predictions
python src/predict.py --input data/test/Fist/sample_14_Fist.png --output outputs/one.png
python src/run.py --input demo.mp4 --output outputs/video_demo
```

## Genuine MediaPipe option — requires a new real-hand dataset

A real landmark extractor is implemented in `src/gesture_runtime.py` (63 normalized landmark features). It **never substitutes synthetic keypoints**. When real labeled images are present in `data/train/<class>/`, run:

```bash
python src/train.py --backend mediapipe --output models/gesture_mediapipe.npz
python src/evaluate.py --split test --weights models/gesture_mediapipe.npz
python src/app.py --weights models/gesture_mediapipe.npz
```

If MediaPipe cannot find all six classes in training, training fails explicitly. Do not train or tune on test images. Save the final verified weights on GitHub Releases or a documented drive link if they grow too large, and note the file SHA-256 in `metrics.json`.

## Repository contents

- `src/gesture_runtime.py` — feature extraction, model loading, inference and rendering (true box/points).
- `src/train.py`, `src/evaluate.py` — train without test leakage; generate confusion matrix and JSON.
- `src/app.py`, `src/demo_stream.py`, `src/run.py`, `src/predict.py` — Gradio, continuous webcam, batch/video and single-image runners.
- `src/baseline.py` — classical color-moments + 1NN baseline.
- `src/test_cases.py`, `outputs/test_cases/` — 12 cases (including >=3 hard cases), actual results, outputs.
- `models/gesture_contour.npz` — compact shipped model weights, backend documented.
- `docs/data_card.md`, `docs/milestone4_report_draft.md`, `docs/remaining.md` — data provenance, CP4 report and outstanding work.
- `docs/final_report.pdf`, `docs/final_slides.pdf`, `docs/contributions/` — final drafts when available.
- `outputs/eval/` — metrics.json, per-image results and confusion matrix.

## Source/licence and privacy

The illustrations in the archive have **no verified external dataset licence**. HaGRID-like filenames or metadata do not prove HaGRID provenance. Collect real-hand examples only with permission; obtain consent and permissions before filming minors and avoid publishing identifying media.

## Team contributions and Git tags

Historical proposals name **Bulos** for preprocessing/baseline and **Ramos** for main classifier/demo, but the provided Git history contains only one initial commit attributed to ThaniaKira. That history **does not verify individual authorship** of each file. Each person must add their own true 3–5 important commit links to the individual statement. Local snapshot tags `cp4-80` / `v1.0` in distributed ZIPs are not proof of a remote push: confirm them on GitHub after review.

Run `git tag -l`, `git log --oneline --decorate`, then `git push origin main --tags` **only after your team reviews the changes and authenticates the commits**. Avoid retagging old grades or falsely attributing contributions.
