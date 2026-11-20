# Verification checklist at packaging time

**Passed (offline and reproducible in the authoring environment):**

1. Compiled updated `src/` Python modules with `compileall`.
2. Importer read a simulated ZIP using the **documented HaGRID ZIP folder layout**, parsed annotations, cropped images, created `manifest.csv`, and separated the same `user_id` across train/validation/test.
3. A complete train/export/load/evaluate smoke test passed on synthetic illustrated images. These synthetic scores **are not reported as real accuracy**.
4. The original, already trained contour NPZ still evaluated on the six illustrated test samples (6/6 accuracy; cannot generalize to webcam).

**Not possible to verify here:**

1. Kaggle real dataset download (the authoring environment has no external binary-download access; expected ZIP ~1GB).
2. Real MediaPipe hand detection and Kaggle-trained Random Forest accuracy (MediaPipe package unavailable in authoring environment; supported by Windows setup requirements).
3. FPS and stability with the student's actual laptop webcam.
4. End-to-end Gradio launch under the student's exact Windows package installation (their previous Gradio dependency issue motivated the added version pins).

**Target of 80–90% real test accuracy: unverified, not achieved by assertion.** The model and metric generation scripts must be executed after the Kaggle dataset is downloaded. Do not alter the held-out test set or manually set reported accuracy to a desired percentage.
