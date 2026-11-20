# Phase 5 · Website UI/UX release notes

## Purpose

This release updates **only the user-facing website** for the Blurry Vision educational hand gesture recognition system. It **does not retrain, replace, tune, or modify the computer vision algorithms or model weights** from the user's uploaded Phase 4 ZIP.

## Pages and navigation

| Browser path | Page | Description |
|---|---|---|
| `/` | Home | Futuristic, pink-purple landing page with an oversized typographic hero and illustrated landmark hand. |
| `/studio` | Recognition Studio | **Separate browser page**, not a section beneath the homepage. Upload, webcam snapshot, recognition overlay, result statistics, and download. |
| `/guide` | Gesture Guide | Six core gestures with the included illustrated sample images and links to load each illustration into the Studio. |

Website has responsive mobile navigation, keyboard accessible controls, drag-and-drop, camera permission handling, and clear error and loading states. CSS and JS are local; no remote fonts, hosted scripts, accounts, or external design services are required.

## Preservation of the existing recognition backend

- **Unmodified**: `models/*`, `src/gesture_runtime.py`, `src/geometry_rules.py`, `src/preprocess.py`, `src/train.py`, `src/evaluate.py`, `src/prepare_kaggle.py`, and the other original evaluation/data files.
- **Updated**: `src/app.py` now starts a FastAPI web application (keeping the same command, `python src/app.py`).
- **New**: `src/site.py` calls the existing `GestureRecognizer` or `GeometryRecognizer`, then the same `draw_result()` helper. This moves image transfer from Gradio components to a small JSON API without changing predictions or preprocessing.
- **New assets**: `web/templates/*.html`, `web/static/css/site.css`, `web/static/js/*.js`, `web/static/img/*.svg`. Guide samples are **copies** of existing project illustrations and are not new training data.

### Model selection is unchanged

1. If `models/gesture_hagrid_mediapipe.npz` exists, load it with existing `GestureRecognizer` (same original auto-selection rule).
2. Otherwise use the existing MediaPipe `GeometryRecognizer` heuristic fallback. This is **not** a newly trained Kaggle model and does **not** have a calibrated confidence score.
3. Optional existing weights can be specified: `python src/app.py --weights models/your_existing_file.npz`. No training is done.

The particular uploaded ZIP contained `models/gesture_contour.npz`, **not** `models/gesture_hagrid_mediapipe.npz`. If you trained a better Kaggle-based model on your laptop after making the uploaded ZIP, copy that unchanged `.npz` file into `models/` with the expected name (or provide its path using `--weights`). The site's status indicator reads the backend from the file. Do not claim a specific accuracy until you reproduce it from your held-out data.

## Windows Quickstart

Open the extracted project folder in VS Code. Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/app.py
```

In a browser, visit:

- Home: http://127.0.0.1:7860/
- Studio: http://127.0.0.1:7860/studio
- Guide: http://127.0.0.1:7860/guide

The server must remain running while you use the website. To stop it, press Ctrl+C in the terminal. If port 7860 is busy, run `python src/app.py --port 7870`.

### Example UI flow

1. Open the Home page and click **Enter the Studio**.
2. **Upload photo** or choose **Use camera** and allow webcam access; capture one frame.
3. Click **Analyze Gesture**. The backend returns the *same* original annotated output and metadata, now displayed in a styled result panel.
4. Download the annotated result if desired.
5. Open **Gesture Guide** to view six illustrated reference poses. Click **Try this pose** to preload an illustration in Studio and analyze it (not a real-world accuracy test).

## Troubleshooting

- **No hand detected**: use better lighting, move the palm completely into view, and use a non-cluttered background.
- **MediaPipe missing**: install the existing dependencies exactly as pinned in `requirements.txt`.
- **Model status shows geometric rules**: the Kaggle-trained model weight file is not in the folder. The UI did not replace it or retrain it.
- **Camera does not open**: in your browser allow webcam permission for `127.0.0.1`; ensure other apps are not locking the camera, or use image upload.
- **Port occupied**: `python src/app.py --port 7870`.
- **Evaluation numbers**: run the existing `src/evaluate.py` with the same model, dataset, and test split used in your report. Interface screenshots do not establish classification accuracy.

## Phase 5 presentation notes

- Show the Home and Guide as evidence of navigation and educational UI design.
- Open `/studio` to demonstrate image upload, webcam frame capture, result overlay, performance metadata, and download.
- Show a successful sample **and a difficult failure case**.
- Report speed on the actual presentation laptop and provide the existing `src/evaluate.py` results from a held-out test set.
- The website screenshots in `outputs/final_ui/` are **interface previews** and not evidence of model accuracy.
- Final report, PDF slides, actual demonstration recording, and individual Git contributions must still reflect genuine completed work and actual evidence.
