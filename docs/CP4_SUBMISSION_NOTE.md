# CP4 hand-in evidence and missing validations

- Local tag: `cp4-80`, created for this updated working copy after the original 12 Nov due date. It **does not prove** that these files existed at the original deadline or were uploaded to GitHub.
- Gradio interface: `python src/app.py`; saved model `models/gesture_contour.npz`.
- Current bundled evaluator: `python src/evaluate.py --split test --weights models/gesture_contour.npz`.
- `metrics.json` and `predictions.csv` use **six original illustrated test images** (one per class).
- 12 documented acceptance/stress cases: `python src/test_cases.py`, currently 8 pass / 4 fail.
- `outputs/backup_illustrated_demo.mp4` is an **offline illustrated results reel**, not a real 5-minute application screen recording; replace it before submission.
- Checkpoint 4 report and slides are draft artifacts. A live teacher-provided input test, laptop FPS and genuine MediaPipe model verification remain uncompleted.
- The source data does not establish actual individual authorship; every member must verify their own Git commits and code.
