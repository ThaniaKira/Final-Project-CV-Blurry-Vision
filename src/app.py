"""Gradio interface (upload or webcam snapshot); use demo_stream.py for continuous FPS."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import cv2
import gradio as gr
import numpy as np
from src.pipeline import run, get_recognizer


def build_app(weights="models/gesture_contour.npz"):
    # Fail early and cache model before user interaction, not on each click.
    model=get_recognizer(weights)

    def infer(rgb):
        if rgb is None:
            return None,{"error":"Upload an image or capture a webcam snapshot"}
        frame=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2BGR)
        result,details=run(frame,weights)
        if model.backend=="contour":
            details["warning"]="Contour demonstration backend trained on 8 illustrated samples. Not a validated MediaPipe model."
        return cv2.cvtColor(result,cv2.COLOR_BGR2RGB),details

    with gr.Blocks(title="Blurry Vision — Gesture Learning") as app:
        gr.Markdown("# Blurry Vision | Educational Hand Gesture Recognition")
        gr.Markdown("Six static gestures: Open Hand, Fist, Peace, Thumbs Up, Pointing, OK. "
                    "Confidence is a random-forest vote fraction, not a calibrated probability. "
                    f"**Active model:** `{model.backend}`")
        with gr.Row():
            image=gr.Image(label="Upload image or capture webcam",type="numpy",sources=["upload","webcam"])
            display=gr.Image(label="Prediction and hand region",type="numpy")
        details=gr.JSON(label="Prediction details, FPS and processing time")
        btn=gr.Button("Recognize gesture",variant="primary")
        btn.click(infer,inputs=image,outputs=[display,details])
        gr.Markdown("For continuous video, run `python src/demo_stream.py --webcam 0`. "
                    "This teaching prototype is not validated for classroom deployment.")
    return app


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--weights",default="models/gesture_contour.npz")
    p.add_argument("--port",type=int,default=7860)
    a=p.parse_args()
    build_app(a.weights).launch(server_name="127.0.0.1",server_port=a.port)
