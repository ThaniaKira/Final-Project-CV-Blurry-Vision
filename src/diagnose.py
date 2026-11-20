"""Non-destructive environment/dataset/model readiness check; python src/diagnose.py"""
import sys
from pathlib import Path
from importlib import import_module

print("Blurry Vision Phase 4 pre-flight")
print("Python:",sys.version.split()[0])
for name in ('numpy','cv2','sklearn','mediapipe','gradio','fastapi','starlette','pydantic','huggingface_hub'):
    try:
        obj=import_module(name)
        print(f"{name:18} {getattr(obj,'__version__','installed')}")
    except Exception as exc:
        print(f"{name:18} NOT AVAILABLE ({exc})")
for path in ('models/gesture_contour.npz','models/gesture_hagrid_mediapipe.npz',
             'data/hagrid_real/data_card.json','data/hagrid_real/manifest.csv'):
    print(path,':', 'PRESENT' if Path(path).is_file() else 'NOT YET GENERATED')
print("For Kaggle training instructions: docs/KAGGLE_TRAINING.md")
