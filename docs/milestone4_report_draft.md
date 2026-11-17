# Milestone 4 — 80% Progress Check | DRAFT (Team Blurry Vision)

**Project:** Real-Time Educational Hand Gesture Recognition for Interactive Early Childhood Learning
**Phase:** CP4 (`cp4-80` target, 12 November 2026)
**Status:** Application and package prepared; **80% completion not independently established**. Real webcam and laptop FPS verification are still required.

## 1. Introduction and objectives
Affordable touchless gesture control could supplement early-childhood learning activities. Our six target signs are Open Hand, Fist, Peace, Thumbs Up, Pointing, and OK. The original proposal targets macro-F1 >=0.85 on held-out images, >=15 FPS on a CPU laptop, and robustness across 150–500 lux. We can currently demonstrate an offline classical contour model on illustrated samples, but cannot yet claim the intended real-camera results.

## 2. Data and ethical limitations
The uploaded repository has 20 stylized drawings, split 8 train / 6 validation / 6 test (see `docs/data_card.md`). These are not human webcam photos; neither source independence nor use of genuine HaGRID photos is substantiated. Do not generalize scores to real students or real classroom lighting. No actual video of a child was collected for this build.

## 3. Algorithm and system implementation
**Image or video input → resize and CLAHE → YCrCb skin segmentation → largest hand-shaped contour → 16×16 normalized silhouette + Hu moments + geometric features → Random Forest (120 trees, seed 42) → class label, vote confidence, true contour box, processing time.** The classical comparison is the original color moments + kNN baseline. These are two **classical** pipelines, and the shipped weights `models/gesture_contour.npz` belong to the contour pipeline, not MediaPipe. An additional `mediapipe` backend is implemented but requires real hands and was not evaluable in this container.

The browser interface is `python src/app.py` (Gradio upload or webcam snapshot). The continuous desktop camera interface is `python src/demo_stream.py --webcam 0`; video files and image folders are supported by `src/run.py`. All use the same packaged classifier rather than retraining on every image.

## 4. Experiments and reproducible test metrics
Run: `python src/evaluate.py --split test --weights models/gesture_contour.npz`. This regenerates `outputs/eval/metrics.json`, `confusion_matrix.png`, and `predictions.csv`. Each of the six gesture classes has exactly one illustration in the test set. In the checked evaluation environment (Linux / Python 3.13 / OpenCV 4.13):

| Model | Accuracy | Macro-F1 | Mean processing ms / image (environment-specific) | Approx. FPS |
|---|---:|---:|---:|---:|
| Original YCrCb moments + 1NN baseline, re-evaluated | 16.67% | 0.0667 | ~2.3 | ~439 |
| **Contour silhouette + Random Forest, packaged CP4 model** | **100% (6/6)** | **1.0000** | **~8.2** | **~122.6** |

The 6/6 result is **not** a generalization claim. The local environment is not the demo laptop. Hardware, runtime and measurement noise change processing times; record demo-laptop speed after warmup and rerun before class. The CP3 report asserted 100% with **MediaPipe 21 landmarks + RF** at 18.92 ms/frame and a 66.67% baseline; those figures are **historical claims, not reproduced here**, and are NOT directly comparable to the contour implementation. The updated baseline is 1/6, which is inconsistent with the historical number. A truthful CP3→CP4 delta requires the original CP3 weights, dependencies and test protocol; these are missing from the ZIP.

## 5. Test-case results and hard cases
`python src/test_cases.py` executes 12 cases and writes CSV, annotated images, pass/fail status. On the current illustrated samples, **8/12 pass**, including all six unaltered test illustrations; edge cases TC07, TC09, TC10 and TC11 fail. Cases TC07–TC11 are **transformed stress probes of already-held-out drawings** and are not additional independent test samples. See `outputs/test_cases/test_cases.csv`.

| ID | Condition | Expected | Actual | Outcome |
|---|---|---|---|---|
| TC07 | Very dim Peace | Peace | No hand detected | FAIL |
| TC08 | Heavily blurred Fist | Fist | Fist | PASS |
| TC09 | Very small Pointing | Pointing | Peace | FAIL |
| TC10 | 40° rotated OK | OK | Open_Hand | FAIL |
| TC11 | Brightness-shifted Thumbs_Up | Thumbs_Up | No hand detected | FAIL |

## 6. Analysis and limitations
The thresholded region disappears under extreme brightness shifts (TC07, TC11). Downscaled gestures have fewer recoverable geometric details (TC09). Large rotation changes silhouette relationships (TC10). The model has only 1–2 training illustrations per class; it is not suitable for clinical use, real child tracking, multiple hands, outdoor use, high motion, or varied real-life camera conditions. Random-forest votes are not calibrated confidence probabilities. The drawn box is an image region from a color mask; it is not a human-body detector. Live webcam behavior has not been tested in this environment, so `>=15 FPS on the presentation laptop` remains **unverified**.

## 7. Completion list and evidence
Created: packaged weights, Gradio UI, image/video/webcam CLI, continuous webcam display, 12-case test matrix with real failures, deterministic evaluation artifact, reproducibility instructions, draft slides, documentation. Remaining before a credible 100% project: collect/validate actual hand data, train the real MediaPipe head, benchmark on demo laptop, test teacher-like unknown inputs, record webcam demonstration, inspect all output screenshots and verify individual commit ownership. See `docs/remaining.md`.

**Accuracy is demonstrated only on the six drawings; no overall project completion percentage is asserted without owners and signed-off task checklist.**
