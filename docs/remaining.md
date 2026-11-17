# CP4 → Final: written task list (teacher sign-off pending)

> These are **proposed**, not agreed in class. Confirm owners and priorities with the instructor. Deadline Thu 19 Nov 2026, 11:30 PM.

| Task | Proposed owner | Due | Evidence of completion |
|---|---|---|---|
| Record actual clean environment setup with Python 3.11 on presentation laptop; verify Gradio starts | Team (assign) | Nov 18 | setup screen + exact `pip freeze` |
| Collect source-separated, consented real hand photo examples across all six gestures and indoor lighting; freeze new test split | Data owner (verify) | Nov 18 | data card, count and source groups |
| Train genuine MediaPipe landmark RF (`python src/train.py --backend mediapipe`) only on train images | Model owner (verify) | Nov 18 | saved weights + hash |
| Re-run `src/evaluate.py` using final genuine model and held-out real-world test set; update report/slides to those generated values | Evaluation owner (verify) | Nov 19 | `metrics.json`, confusion matrix and SHA |
| Measure video throughput and model load time on exact demo laptop, warm-up then 30+ frames | Demo owner (verify) | Nov 19 | benchmark log with laptop specs |
| Verify 10+ true test cases including >=3 difficult camera cases, not only synthetic transformations | Testing owner (verify) | Nov 19 | updated `test_cases.csv` and actual outputs |
| Capture <=5-minute real app/webcam screen recording, replacing the illustrated offline backup | Demo owner (verify) | Nov 19 | `outputs/final/demo_recording.mp4` |
| Check final report facts, citations and individual contributions against Git history; export final slides to PDF | All members | Nov 19 | final PDFs + contribution table |
| Rehearse with teacher-like unseen inputs, power cables, camera permissions and projector | All members | Nov 19 | rehearsal checklist signed |
| Commit actual verified changes under correct developers, tag and push `cp4-80` / `v1.0` appropriately | Repo owner | Nov 19 | public tag links and SHA |
