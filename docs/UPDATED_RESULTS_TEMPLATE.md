# Phase 4 — Kaggle real-photo evaluation (fill ONLY after training)

**Dataset:** HaGRID Sample 30k 384p, Kaggle `innominate817/hagrid-sample-30k-384p`.

**Recognition model:** MediaPipe 21 landmarks + Random Forest from `models/gesture_hagrid_mediapipe.npz`.

**Data partitions:** See `data/hagrid_real/data_card.json` and `manifest.csv`. Train/validation/test subject IDs must not overlap.

| Reproducible measurement | Result after running actual command |
|---|---|
| Train / validation / test photos | [read `data_card.json`] |
| Test accuracy | [read `outputs/hagrid_eval/metrics.json`] |
| Test macro-F1 | [read `outputs/hagrid_eval/metrics.json`] |
| Hand-detection rate | [read `outputs/hagrid_eval/metrics.json`] |
| Main model ms/image, FPS | [run on the actual demo laptop] |
| Per-class precision, recall, F1 | [read metrics.json] |
| Misclassifications and contributing factors | [inspect predictions.csv and image gallery] |
| Model SHA256 | [read metrics.json] |

**No values in this template have been invented.** If the 80–90% target is missed, explain why and demonstrate a validation-based improvement plan. The 100% figure previously claimed on six illustrated images is not comparable to a held-out HaGRID test score.
