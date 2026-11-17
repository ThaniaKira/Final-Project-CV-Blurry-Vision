"""Batch images, image, video or webcam. Same model/pipeline as Gradio and evaluate.py."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import csv
from pathlib import Path
import cv2

from src.gesture_runtime import GestureRecognizer, draw_result

IMAGE_EXT={".jpg",".jpeg",".png",".bmp",".webp"}
VIDEO_EXT={".mp4",".avi",".mov",".mkv"}


def process_image(path, output_dir, model):
    image=cv2.imread(str(path))
    if image is None:
        print("Skipping unreadable image:",path)
        return None
    pred=model.predict(image)
    relative=Path(path).name
    save=output_dir/("res_"+relative)
    cv2.imwrite(str(save),draw_result(image,pred))
    print(f"{path}: {pred.label} ({pred.confidence:.2f}), {pred.latency_ms:.1f} ms")
    return [str(path),pred.label,pred.confidence,round(pred.latency_ms,2),str(save)]


def process_video(source, output_dir, model, webcam=False, max_frames=0):
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video or webcam: {source}")
    writer=None
    count=0
    try:
        while cap.isOpened():
            ok, frame=cap.read()
            if not ok: break
            pred=model.predict(frame)
            annotated=draw_result(frame,pred)
            if webcam:
                cv2.imshow("Blurry Vision — press Q to quit",annotated)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            else:
                if writer is None:
                    h,w=annotated.shape[:2]
                    fps=cap.get(cv2.CAP_PROP_FPS) or 20.0
                    writer=cv2.VideoWriter(str(output_dir/"annotated_video.mp4"),
                                           cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
                writer.write(annotated)
            count+=1
            if max_frames and count>=max_frames: break
    finally:
        cap.release()
        if writer is not None:writer.release()
        if webcam:cv2.destroyAllWindows()
    print(f"Processed {count} frames" + ("; saved annotated_video.mp4" if not webcam else ""))


def run(input_path=None,output="outputs/run_output",webcam=None,weights="models/gesture_contour.npz",max_frames=0):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=GestureRecognizer(weights,stream=webcam is not None or (input_path and Path(input_path).suffix.lower() in VIDEO_EXT))
    try:
        if webcam is not None:
            process_video(webcam,output,model,webcam=True,max_frames=max_frames)
        elif input_path is None:
            raise ValueError("Provide --input or --webcam")
        else:
            p=Path(input_path)
            if p.is_dir():
                all_files=sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
                if not all_files:raise RuntimeError(f"No images found in {p}")
                rows=[process_image(f,output,model) for f in all_files]
                with open(output/"predictions.csv","w",newline="",encoding="utf-8") as f:
                    writer=csv.writer(f);writer.writerow(["file","gesture","confidence","processing_ms","output_file"])
                    writer.writerows(row for row in rows if row)
            elif p.suffix.lower() in IMAGE_EXT:
                process_image(p,output,model)
            elif p.suffix.lower() in VIDEO_EXT:
                process_video(str(p),output,model,max_frames=max_frames)
            else:
                raise ValueError("Unknown input; use an image, image directory or video")
    finally:
        model.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default=None)
    p.add_argument("--output",default="outputs/run_output")
    p.add_argument("--webcam",type=int,default=None)
    p.add_argument("--weights",default="models/gesture_contour.npz")
    p.add_argument("--max-frames",type=int,default=0)
    a=p.parse_args()
    run(a.input,a.output,a.webcam,a.weights,a.max_frames)
