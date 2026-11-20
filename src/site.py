"""Phase 5 website interface. The original CV feature extraction and models are untouched.

Separate browser routes: / (home), /studio (recognition), /guide (gesture reference).
API passes images to the original GestureRecognizer / GeometryRecognizer and draw_result.
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.geometry_rules import GeometryRecognizer
from src.gesture_runtime import BackendUnavailable, GestureRecognizer, draw_result

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DEFAULT_WEIGHTS = ROOT / "models" / "gesture_hagrid_mediapipe.npz"
# Only choose a trained model automatically when its actual file exists.
WEIGHTS = Path(os.environ.get("BLURRY_VISION_WEIGHTS", str(DEFAULT_WEIGHTS)))
if not WEIGHTS.is_absolute():
    WEIGHTS = (ROOT / WEIGHTS).resolve()
LOCK = Lock()  # Reuse and serialize access to the same MediaPipe detector instance.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def mode():
    return "trained" if WEIGHTS.exists() else "rules"


@lru_cache(maxsize=1)
def recognizer():
    if WEIGHTS.exists():
        return GestureRecognizer(WEIGHTS)
    return GeometryRecognizer()


def status():
    trained = WEIGHTS.exists()
    backend = "rules"
    if trained:
        try:
            with np.load(WEIGHTS, allow_pickle=False) as archive:
                backend = str(archive["backend"].item())
        except (OSError, KeyError, ValueError):
            backend = "unknown"
    if backend == "mediapipe":
        label = "Saved MediaPipe + Random Forest"
        note = "Loaded the existing MediaPipe model weights, unchanged. Evaluate accuracy separately on held-out real photos."
    elif backend == "contour":
        label = "Saved Contour + Random Forest"
        note = "Loaded existing contour model weights, unchanged. These are not validated on real webcam photos."
    elif trained:
        label = "Saved recognition model"
        note = "Existing model weights were detected. Verify the backend and held-out evaluation."
    else:
        label = "MediaPipe · Geometry rules"
        note = "No trained MediaPipe weights were found in this ZIP. The existing geometric-rules fallback is active."
    return {
        "mode": "trained" if trained else "rules",
        "backend": backend,
        "label": label,
        "has_trained_weights": trained,
        "model_path": WEIGHTS.relative_to(ROOT).as_posix() if WEIGHTS.is_relative_to(ROOT) else str(WEIGHTS),
        "note": note,
    }


app = FastAPI(title="Blurry Vision — Gesture Recognition", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


@app.get("/", include_in_schema=False)
async def homepage():
    return FileResponse(WEB / "templates" / "index.html")


@app.get("/studio", include_in_schema=False)
@app.get("/studio/", include_in_schema=False)
async def studio():
    return FileResponse(WEB / "templates" / "studio.html")


@app.get("/guide", include_in_schema=False)
@app.get("/guide/", include_in_schema=False)
async def guide():
    return FileResponse(WEB / "templates" / "guide.html")


@app.get("/api/status")
async def api_status():
    return status()


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    # Only pixels enter the existing pipeline; uploaded media is not written to disk.
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Please upload or capture an image first.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 12 MB. Try a smaller JPEG or PNG.")
    try:
        npbuf = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    except (ValueError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail="Could not decode the image.") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image. Use JPG, PNG or WebP.")
    h, w = frame.shape[:2]
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum is 16 megapixels.")

    try:
        with LOCK:
            prediction = recognizer().predict(frame)
            rendered = draw_result(frame, prediction)
    except (ImportError, ModuleNotFoundError, BackendUnavailable) as exc:
        raise HTTPException(status_code=503, detail="The active model requires MediaPipe. Install requirements.txt and restart the app.") from exc
    except Exception as exc:
        # Do not return raw stack traces or file paths to the browser.
        raise HTTPException(status_code=500, detail="Recognition failed. Check the VS Code terminal for details.") from exc

    ok, png = cv2.imencode(".png", rendered)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to render recognition output.")
    details = prediction.to_dict()
    if not WEIGHTS.exists():
        details["confidence"] = None
        details["model_status"] = "Untrained geometric-rules fallback; no calibrated confidence or Kaggle accuracy claim."
    else:
        details["model_status"] = "Existing saved model; compare only with the matching evaluation run."
    return JSONResponse({
        "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
        "details": details,
        "status": status(),
    }, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
