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

from src.dual_models import DualRecognizer
from src.gesture_runtime import BackendUnavailable, draw_result

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DEFAULT_WEIGHTS = ROOT / "models" / "gesture_hagrid_mediapipe.npz"
# Only choose a trained model automatically when its actual file exists.
WEIGHTS = Path(os.environ.get("BLURRY_VISION_WEIGHTS", str(DEFAULT_WEIGHTS)))
if not WEIGHTS.is_absolute():
    WEIGHTS = (ROOT / WEIGHTS).resolve()
NUMBER_WEIGHTS = Path(os.environ.get("BLURRY_VISION_NUMBER_WEIGHTS", str(ROOT / "models" / "numbers_hagrid_mediapipe.npz")))
if not NUMBER_WEIGHTS.is_absolute():
    NUMBER_WEIGHTS = (ROOT / NUMBER_WEIGHTS).resolve()
LOCK = Lock()  # Reuse and serialize access to the same MediaPipe detector instance.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def mode():
    return "trained" if WEIGHTS.exists() else "rules"


@lru_cache(maxsize=1)
def recognizer():
    return DualRecognizer(WEIGHTS, NUMBER_WEIGHTS)


@lru_cache(maxsize=1)
def live_recognizer():
    """Independent tracking-mode detector for ordered webcam frames."""
    return DualRecognizer(WEIGHTS, NUMBER_WEIGHTS, stream=True)


@lru_cache(maxsize=1)
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
        label = "Saved MediaPipe + HaGRID ExtraTrees"
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
        "number_model": {
            "mode": "trained" if NUMBER_WEIGHTS.exists() else "rules",
            "weights_present": NUMBER_WEIGHTS.exists(),
            "model_path": NUMBER_WEIGHTS.relative_to(ROOT).as_posix() if NUMBER_WEIGHTS.is_relative_to(ROOT) else str(NUMBER_WEIGHTS),
            "note": ("HaGRID number-classifier weights present; verify held-out test results." if NUMBER_WEIGHTS.exists() else
                     "Untrained finger-counting rules; not a HaGRID model and no measured accuracy claim."),
        },
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


@app.get("/play", include_in_schema=False)
@app.get("/play/", include_in_schema=False)
async def play_and_learn():
    return FileResponse(WEB / "templates" / "play.html")


@app.get("/api/status")
async def api_status():
    return status()


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    # Preserve the original six-gesture website contract and appearance.
    return await _analyze(image, numeric_only=False)


@app.post("/api/analyze/numbers")
async def analyze_numbers(image: UploadFile = File(...)):
    return await _analyze(image, numeric_only=True)


@app.post("/api/live")
async def live_analysis(image: UploadFile = File(...), mode: str = "core"):
    """Compact automatic webcam inference: 0 image rendering/encoding overhead.

    The browser sends occasional JPEG frames from its live camera preview;
    no frames are persisted. Ordered requests reuse the tracking-mode detector.
    """
    if mode not in ("core", "numbers"):
        raise HTTPException(status_code=422, detail="mode must be core or numbers")
    return await _analyze(image, numeric_only=(mode == "numbers"), live=True, include_image=False)


async def _analyze(image: UploadFile, numeric_only=False, live=False, include_image=True):
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
            active = live_recognizer() if live else recognizer()
            core_prediction, number_prediction = active.predict_both(frame)
            detection_attempt = getattr(active.extractor, "last_detection_attempt", "unknown")
            prediction = number_prediction if numeric_only else core_prediction
            rendered = draw_result(frame, prediction) if include_image else None
    except (ImportError, ModuleNotFoundError, BackendUnavailable) as exc:
        raise HTTPException(status_code=503, detail="The active model requires MediaPipe. Install requirements.txt and restart the app.") from exc
    except Exception as exc:
        # Do not return raw stack traces or file paths to the browser.
        raise HTTPException(status_code=500, detail="Recognition failed. Check the VS Code terminal for details.") from exc

    image_uri = None
    if include_image:
        ok, png = cv2.imencode(".png", rendered)
        if not ok:
            raise HTTPException(status_code=500, detail="Unable to render recognition output.")
        image_uri = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    details = prediction.to_dict()
    if live:
        details["points"] = prediction.points  # 21 landmarks for the browser overlay
    details["detection_attempt"] = detection_attempt
    details["number_prediction"] = {
        "gesture": number_prediction.label,
        "number": int(number_prediction.label.split("_")[1]) if number_prediction.label in {f"Number_{i}" for i in range(6)} else None,
        "confidence": number_prediction.confidence if NUMBER_WEIGHTS.exists() else None,
        "method": "trained" if NUMBER_WEIGHTS.exists() else "untrained_geometry_rules",
    }
    if not (NUMBER_WEIGHTS.exists() if numeric_only else WEIGHTS.exists()):
        details["confidence"] = None
        details["model_status"] = "Untrained MediaPipe geometric rules; no HaGRID accuracy claim."
    else:
        details["model_status"] = "Saved HaGRID weights loaded; consult separate held-out evaluation for this head."
    result = {"details": details, "status": status()}
    if include_image:
        result["image"] = image_uri
    return JSONResponse(result, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
