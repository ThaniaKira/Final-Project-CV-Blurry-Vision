# Blurry Vision Phase 4 — update and evaluation status

## Verified prior limitation

Observed webcam snapshot: the model predicted `Pointing` for a user showing an open hand. It reported a weak Random Forest vote of approximately **0.2833**, and the detected skin contour enclosed part of the torso/shoulder instead of the fingers. The 8-image cartoon-trained contour model was not suitable for real webcam photos.

## Added or revised implementation

| Component | File | Purpose |
|---|---|---|
| Source-subject disjoint dataset importer | `src/prepare_kaggle.py` | Import six aligned HaGRID classes from a real Kaggle dataset; crop true annotated hand boxes; split by person |
| Genuine landmark extraction | `src/gesture_runtime.py` | MediaPipe 21 hand landmarks on native-aspect-ratio RGB images, normalized by wrist-to-palm span |
| Six-class Random Forest model training | `src/train.py` | Select parameters using validation only; portable `.npz`; training-only mirroring |
| Real-hand rule baseline | `src/geometry_rules.py` | Immediate demo before Kaggle training; rule predictions intentionally have no model probability |
| Updated web interface | `src/app.py` | Select rule / genuinely Kaggle-trained RF / legacy illustration model; webcam snapshot |
| Continuous webcam window | `src/demo_stream.py` | Real hand landmarks + predicted label (CPU FPS visible) |
| Real held-out evaluation | `src/evaluate.py` | Accuracy, macro-F1, per-class metrics, confusion matrix, detection rate, sample speed; subject separation audit |
| Windows setup instructions | `docs/KAGGLE_TRAINING.md` | Download, prepare, train, run, evaluate |

## What remains unverified in this delivered ZIP

At the time of packaging, Kaggle's approximately 1 GB sample could not be downloaded from this execution environment, so **real photographs, trained Kaggle NPZ weights, and measured Kaggle test metrics are not embedded**. An 80–90% real-hand accuracy result cannot be claimed until those steps actually run. The prior illustrations are retained only for comparison.

After training, add values from `outputs/hagrid_eval/metrics.json` to the phase 4 report and replace the original six-illustration evidence with real photographic evidence. Do not claim all evaluation numbers reproduce unless the exact weights, data source and split manifest are present in the submitted release.

## Student demonstration checklist

- Show a High Five / open hand; the MediaPipe box should encircle actual fingers.
- Try Fist, Peace, Thumbs Up, Pointing, and OK, one gesture at a time.
- Show at least three adverse-condition images and record expected/actual.
- Execute `python src/evaluate.py --split test --data data/hagrid_real --weights models/gesture_hagrid_mediapipe.npz --output outputs/hagrid_eval` in front of the instructor.
- Explain how `user_id` prevents people from appearing in both the training and test partitions.
- State honestly if a real landmark model is still untrained or fails the desired metric.
