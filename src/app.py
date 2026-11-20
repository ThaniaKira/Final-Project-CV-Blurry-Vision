"""Blurry Vision's upload-based gesture-recognition browser experience."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.gesture_runtime import GestureRecognizer, draw_result
from src.geometry_rules import GeometryRecognizer

KAGGLE_WEIGHTS = Path("models/gesture_hagrid_mediapipe.npz")

CSS = """
:root { --bg:#09070f; --panel:rgba(35,25,56,.52); --panel-soft:rgba(51,36,80,.38); --line:rgba(216,190,255,.27); --text:#f7f2ff; --muted:#b7adca; --accent:#aa6cff; --accent-light:#dcc3ff; }
body, .gradio-container { background:radial-gradient(circle at 13% 18%,rgba(142,69,255,.26),transparent 24%),radial-gradient(circle at 87% 78%,rgba(191,88,255,.20),transparent 28%),var(--bg) !important; color:var(--text) !important; }
.gradio-container { max-width:none !important; min-height:100vh; margin:0 !important; padding:0 !important; overflow-y:auto; font-family:Arial,Helvetica,sans-serif !important; }
.landing-shell { min-height:100vh; display:flex; justify-content:center; align-items:center; padding:28px .5in !important; box-sizing:border-box; }.landing-shell > * { width:100%; }
.landing-card { width:min(100%,1180px); min-height:720px; margin:0 auto; padding:clamp(42px,6vw,92px); box-sizing:border-box; border:1px solid var(--line); border-radius:28px; background:radial-gradient(circle at 76% 32%,rgba(160,92,255,.23),transparent 22%),linear-gradient(125deg,rgba(46,29,73,.70),rgba(11,8,18,.66) 62%); backdrop-filter:blur(18px); display:flex; flex-direction:column; justify-content:center; text-align:left; box-shadow:0 30px 80px rgba(0,0,0,.36),0 0 64px rgba(137,69,255,.13); }
.site-header { width:min(100%,1180px); margin:0 auto 18px; display:flex; justify-content:space-between; align-items:center; color:#f7f2ff; }.site-logo { font-size:1.02rem; font-weight:850; letter-spacing:.08em; }.site-logo span { color:#d8bcff; }.site-nav { display:flex; gap:24px; color:#beb2cf; font-size:.84rem; }.site-nav span:first-child { color:#fff; }.site-footer { width:min(100%,1180px); margin:18px auto 0; padding:18px 0 2px; border-top:1px solid var(--line); color:#a99abf; font-size:.82rem; letter-spacing:.04em; }
.eyebrow { color:var(--accent-light); font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; font-weight:700; }.landing h1 { max-width:790px; font-size:clamp(3.5rem,7.5vw,7.6rem); line-height:.91; letter-spacing:-.075em; margin:18px 0 24px; color:var(--text); font-weight:700; }.landing h1 span { color:var(--accent-light); }.landing p { max-width:550px; color:var(--muted); font-size:1.06rem; line-height:1.7; margin:0; }.capabilities { display:flex; flex-wrap:wrap; gap:10px; margin:32px 0; }.capabilities span { border:1px solid var(--line); border-radius:999px; padding:9px 13px; color:#ddd1f0; background:rgba(189,140,255,.06); font-size:.82rem; }
.primary-cta { width:240px; margin:0 !important; }.primary-cta button { min-height:56px !important; width:100%; border:1px solid rgba(231,211,255,.52) !important; border-radius:14px !important; background:linear-gradient(120deg,rgba(152,87,238,.82),rgba(198,124,255,.76)) !important; box-shadow:0 13px 30px rgba(132,65,234,.32),0 0 24px rgba(190,123,255,.25),inset 0 1px rgba(255,255,255,.25); color:#fff !important; font-size:.98rem !important; font-weight:700 !important; letter-spacing:.01em; }.primary-cta button:hover { filter:brightness(1.12); transform:translateY(-1px); }.landing-shell .notice { width:min(100%,1180px); margin:15px auto 0; text-align:left; padding-left:clamp(42px,6vw,92px); box-sizing:border-box; }
.hero-actions { display:flex !important; gap:14px; align-items:center; width:min(100%,1180px); margin:-145px auto 105px; padding-left:clamp(42px,6vw,92px); box-sizing:border-box; position:relative; z-index:2; }.primary-cta { width:188px; }.primary-cta button { min-height:50px !important; border-radius:999px !important; background:linear-gradient(180deg,#f8efff,#b87cef) !important; color:#251132 !important; box-shadow:0 12px 27px rgba(176,98,246,.32),inset 0 1px rgba(255,255,255,.96) !important; }.secondary-cta { width:156px; }.secondary-cta button { min-height:50px !important; border:1px solid rgba(235,215,255,.65) !important; border-radius:999px !important; background:rgba(34,20,50,.32) !important; color:#f3e9ff !important; font-weight:650 !important; }
.workspace-shell { width:100%; min-height:100vh; background:linear-gradient(135deg,#0d0915,#171024); }.topbar { display:flex; align-items:center; justify-content:space-between; padding:22px .5in; border-bottom:1px solid var(--line); background:rgba(14,10,23,.48); backdrop-filter:blur(16px); }.brand { font-size:1.05rem; font-weight:800; letter-spacing:.08em; color:var(--text); }.brand i { color:var(--accent-light); font-style:normal; }.back button { min-height:40px !important; border:1px solid var(--line) !important; border-radius:10px !important; background:rgba(184,127,255,.08) !important; color:#e9ddfa !important; }.studio-footer { width:min(calc(100% - 1in),1180px); margin:0 auto; padding:18px 0 26px; border-top:1px solid var(--line); color:#a99abf; font-size:.82rem; letter-spacing:.04em; }
.studio-content { width:min(calc(100% - 1in),1180px); margin:0 auto; padding:clamp(44px,6vw,84px) 0 60px; }.studio-lead { display:flex; align-items:end; justify-content:space-between; gap:30px; margin-bottom:34px; }.workspace-title h2 { margin:12px 0 0 !important; color:var(--text); font-size:clamp(2.6rem,5vw,5.2rem) !important; letter-spacing:-.065em; line-height:.95; }.workspace-title p { color:var(--muted); font-size:1rem; line-height:1.6; margin:15px 0 0; }.studio-tag { border:1px solid var(--line); border-radius:999px; color:#e3d2fa; background:rgba(181,120,255,.08); padding:9px 13px; font-size:.74rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; white-space:nowrap; }
.workspace-shell .gr-row { gap:22px; }.workspace-shell .gr-panel, .workspace-shell .form { border:1px solid var(--line) !important; border-radius:18px !important; background:var(--panel) !important; backdrop-filter:blur(14px); box-shadow:0 18px 45px rgba(0,0,0,.16),0 0 38px rgba(132,70,230,.07) !important; }.workspace-shell label span, .workspace-shell .gr-form label { color:#eee5fa !important; }.recognize button { min-height:53px !important; border:1px solid rgba(231,211,255,.52) !important; border-radius:12px !important; background:linear-gradient(120deg,rgba(152,87,238,.85),rgba(198,124,255,.78)) !important; box-shadow:0 13px 30px rgba(132,65,234,.30),0 0 22px rgba(190,123,255,.20) !important; color:white !important; font-size:.98rem !important; font-weight:700 !important; }.notice { color:#aea3bf; font-size:.84rem; line-height:1.5; }.notice b { color:#e1cafa; }.studio-note { max-width:740px; margin:23px auto 0; text-align:center; }
.retry button { min-height:46px !important; border:1px solid var(--line) !important; border-radius:12px !important; background:rgba(180,123,255,.10) !important; color:#eadfff !important; font-size:.94rem !important; font-weight:650 !important; }.notice { color:#aea3bf; font-size:.84rem; line-height:1.5; }.notice b { color:#e1cafa; }.studio-note { max-width:740px; margin:23px auto 0; text-align:center; }
@media(max-width:700px) { .landing-shell{padding:14px !important}.landing-card{min-height:calc(100vh - 28px);padding:38px 26px;border-radius:20px}.landing h1{font-size:3.65rem}.landing-shell .notice{padding-left:26px}.site-nav{display:none}.hero-actions{margin:-110px auto 78px;padding-left:26px;gap:10px}.primary-cta{width:170px}.secondary-cta{width:138px}.topbar{padding:18px 22px}.studio-content{width:calc(100% - 36px);padding-top:45px}.studio-lead{align-items:flex-start;flex-direction:column;gap:18px}.workspace-title h2{font-size:3rem !important;} }
"""


def build_app():
    cached = {}
    active_mode = (
        "Kaggle HaGRID trained: MediaPipe + Random Forest"
        if KAGGLE_WEIGHTS.exists()
        else "MediaPipe hand landmarks + geometric rules"
    )

    def get_model():
        if active_mode not in cached:
            cached[active_mode] = (
                GestureRecognizer(KAGGLE_WEIGHTS)
                if active_mode.startswith("Kaggle")
                else GeometryRecognizer()
            )
        return cached[active_mode]

    def infer(rgb):
        if rgb is None:
            return None, {"error": "Upload an image before analyzing it."}
        frame = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        prediction = get_model().predict(frame)
        details = prediction.to_dict()
        if active_mode.startswith("Kaggle"):
            details["model_status"] = "Trained on real HaGRID photographs. Verify held-out test metrics before reporting accuracy."
        else:
            details["model_status"] = "MediaPipe detects real hand landmarks; gesture names are geometric rules, not trained labels."
            details["confidence"] = None
        return cv2.cvtColor(draw_result(frame, prediction), cv2.COLOR_BGR2RGB), details

    def reset_studio():
        return None, None, None

    show_workspace = lambda: (gr.update(visible=False), gr.update(visible=True))
    show_landing = lambda: (gr.update(visible=True), gr.update(visible=False))
    with gr.Blocks(title="Blurry Vision | Hand Gesture Recognition", css=CSS, theme=gr.themes.Base()) as app:
        with gr.Column(visible=True, elem_classes="landing-shell") as landing:
            gr.HTML("<header class='site-header'><div class='site-logo'>BLURRY <span>VISION</span></div><nav class='site-nav'><span>Home</span><span>Recognition</span><span>Gesture guide</span></nav></header>")
            gr.HTML("""<section class='landing'><div class='landing-card'><div class='eyebrow'>Gesture recognition, reimagined</div><h1>Read the <span>gesture.</span><br>See the intent.</h1><p>A refined vision workspace for recognizing everyday hand gestures from an uploaded image or camera capture.</p><div class='capabilities'><span>21 hand landmarks</span><span>11 gesture poses</span><span>Private local processing</span></div></div></section>""")
            with gr.Row(elem_classes="hero-actions"):
                enter = gr.Button("Explore the studio", variant="primary", elem_classes="primary-cta")
            gr.HTML("<p class='notice'>For dependable gesture classification, select the <b>HaGRID trained model</b> after training and evaluating it on real, unseen photos.</p>")
            gr.HTML("<footer class='site-footer'>Team Blurry Vision 2026</footer>")
        with gr.Column(visible=False, elem_classes="workspace-shell") as workspace:
            with gr.Row(elem_classes="topbar"):
                gr.HTML("<div class='brand'>BLURRY <i>VISION</i></div>")
                back = gr.Button("Home", elem_classes="back")
            with gr.Column(elem_classes="studio-content"):
                gr.HTML("<div class='studio-lead'><div class='workspace-title'><div class='eyebrow'>Computer Vision / 01</div><h2>Recognition studio</h2><p>Bring a gesture into focus. Upload a clear image or capture one with your camera.</p></div><div class='studio-tag'>Image analysis</div></div>")
                gr.HTML(f"<p class='notice'><b>Active model:</b> {active_mode}</p>")
                with gr.Row():
                    picture = gr.Image(label="Your hand image", type="numpy", sources=["webcam", "upload"])
                    result = gr.Image(label="Detected hand and gesture", type="numpy")
                button = gr.Button("Analyze gesture", variant="primary", elem_classes="recognize")
                retry = gr.Button("Try another gesture", elem_classes="retry")
                details = gr.JSON(label="Detection details")
                gr.HTML("<p class='notice studio-note'>For the clearest result, use good front lighting, keep one hand in frame, and avoid heavy motion blur. A “no hand” result means landmarks were not confidently found.</p>")
                button.click(infer, inputs=[picture], outputs=[result, details])
                retry.click(reset_studio, outputs=[picture, result, details])
            gr.HTML("<footer class='studio-footer'>Team Blurry Vision 2026</footer>")
        enter.click(show_workspace, outputs=[landing, workspace])
        back.click(show_landing, outputs=[landing, workspace])
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Optional local port.")
    args = parser.parse_args()
    build_app().launch(server_name="127.0.0.1", server_port=args.port, share=False)
