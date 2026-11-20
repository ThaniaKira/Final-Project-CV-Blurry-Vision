# PHASE 4 SUBMISSION NOTICE — IMPORTANT KAGGLE UPDATE

The files in `docs/milestone4_report_draft.pdf`, `docs/milestone4_slides_draft.pdf`, and older `outputs/eval` are **archives from the original illustration-only checkpoint**, not the Kaggle real-hand training results. They must not be presented as proof of the upgraded model's accuracy.

For enhanced Phase 4, use `docs/CP4_KAGGLE_UPGRADE.md`, `docs/KAGGLE_TRAINING.md`, and `docs/UPDATED_RESULTS_TEMPLATE.md`. Complete the Kaggle download, real model training, held-out evaluation, demo laptop FPS, and 12-case real webcam testing before final submission.

The original `.git` history is preserved in the ZIP, but the enhanced files are **uncommitted modifications**. Create an authentic commit from your Git account and tag the exact tested commit. Do not overwrite prior checkpoint tags as if these upgrades had existed at the earlier deadline.

---

# CP4 hand-in evidence and missing validations

- Local tag: `cp4-80`, created for this updated working copy after the original 12 Nov due date. It **does not prove** that these files existed at the original deadline or were uploaded to GitHub.
- Gradio interface: `python src/app.py`; saved model `models/gesture_contour.npz`.
- Current bundled evaluator: `python src/evaluate.py --split test --weights models/gesture_contour.npz`.
- `metrics.json` and `predictions.csv` use **six original illustrated test images** (one per class).
- 12 documented acceptance/stress cases: `python src/test_cases.py`, currently 8 pass / 4 fail.
- `outputs/backup_illustrated_demo.mp4` is an **offline illustrated results reel**, not a real 5-minute application screen recording; replace it before submission.
- Checkpoint 4 report and slides are draft artifacts. A live teacher-provided input test, laptop FPS and genuine MediaPipe model verification remain uncompleted.
- The source data does not establish actual individual authorship; every member must verify their own Git commits and code.
