import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import argparse
import time
import cv2
from src.preprocess import preprocess_pipeline
from src.main_model import MediaPipeGestureClassifier
from src.data_loader import GestureDataLoader
from src.utils import draw_prediction_overlay

def run_pipeline(input_arg, output_dir="outputs/run_output", webcam_id=None):
    """End-to-end pipeline runner supporting image, directory, video, or webcam stream."""
    loader = GestureDataLoader()
    train_imgs, train_lbls, _ = loader.load_split("train")
    if len(train_imgs) == 0:
        train_imgs, train_lbls, _ = loader.load_split("samples")
        
    model = MediaPipeGestureClassifier()
    model.fit(train_imgs, train_lbls)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Mode 1: Directory input
    if input_arg and os.path.isdir(input_arg):
        image_files = sorted(glob.glob(os.path.join(input_arg, "*.*")))
        print(f"Processing directory '{input_arg}' with {len(image_files)} files...")
        for img_p in image_files:
            if not img_p.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image = cv2.imread(img_p)
            if image is None:
                continue
            resized, _, _, blur_score = preprocess_pipeline(image)
            label, confidence, landmarks = model.predict_single(resized)
            annotated = draw_prediction_overlay(resized, label, confidence, blur_score, landmarks)
            
            out_fname = "res_" + os.path.basename(img_p)
            cv2.imwrite(os.path.join(output_dir, out_fname), annotated)
        print(f"✅ Directory processing complete! Annotated outputs saved to: {output_dir}")

    # Mode 2: Single image input
    elif input_arg and os.path.isfile(input_arg) and input_arg.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(input_arg)
        resized, _, _, blur_score = preprocess_pipeline(image)
        label, confidence, landmarks = model.predict_single(resized)
        annotated = draw_prediction_overlay(resized, label, confidence, blur_score, landmarks)
        out_p = os.path.join(output_dir, "result_" + os.path.basename(input_arg))
        cv2.imwrite(out_p, annotated)
        print(f"✅ Single file processed! Output saved to: {out_p}")

    # Mode 3: Video file or Webcam stream
    elif webcam_id is not None or (input_arg and input_arg.lower().endswith(('.mp4', '.avi', '.mov'))):
        source = webcam_id if webcam_id is not None else input_arg
        print(f"Opening video/webcam stream: {source}...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Unable to open video stream source: {source}")
            return
            
        frame_count = 0
        fps_timer = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 30:
                break
            resized, _, _, blur_score = preprocess_pipeline(frame)
            label, confidence, landmarks = model.predict_single(resized)
            annotated = draw_prediction_overlay(resized, label, confidence, blur_score, landmarks)
            frame_count += 1
            
        cap.release()
        elapsed = time.time() - fps_timer
        fps = frame_count / elapsed if elapsed > 0 else 30.0
        print(f"✅ Stream completed! Processed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Hand Gesture Pipeline (Milestone 3)")
    parser.add_argument("--input", default=None, help="Path to input image, folder, or video file.")
    parser.add_argument("--output", default="outputs/run_output", help="Output directory path.")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0).")
    args = parser.parse_args()
    
    if args.input is None and args.webcam is None:
        run_pipeline("data/samples", args.output)
    else:
        run_pipeline(args.input, args.output, args.webcam)
