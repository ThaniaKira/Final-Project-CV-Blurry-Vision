"""Launch Blurry Vision's Phase 5 two-page website from the original project folder.

    python src/app.py
    python src/app.py --port 7870
    python src/app.py --weights models/gesture_hagrid_mediapipe.npz

The existing model implementation and trained weights remain unchanged.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start Blurry Vision's Phase 5 website")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--weights", default=None, help="Optional path to existing trained weights; never retrains")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.weights:
        os.environ["BLURRY_VISION_WEIGHTS"] = str(Path(args.weights).resolve())
    import uvicorn
    uvicorn.run("src.site:app", host="127.0.0.1", port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
