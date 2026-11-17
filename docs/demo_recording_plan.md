# ≤5-minute backup demo plan

`outputs/backup_illustrated_demo.mp4` is an automated 60–75-second **illustrated test-output reel**. It is NOT a recording of a live webcam run. Replace it with a real screen recording of the working application on your team laptop before submission.

**0:00–0:30** Launch from fresh clone: `python src/app.py`, explain six classes and backend being used.  
**0:30–1:45** Upload one unseen-from-training test image; show class label, confidence (RF vote fraction), contour bounding box, processing time.  
**1:45–3:00** Run cases TC07, TC09, TC10 and explain actual misclassifications and why thresholds/scale change the features.  
**3:00–4:00** Use `python src/demo_stream.py --webcam 0`, changing background and lighting; document measured FPS on demo laptop.  
**4:00–4:45** Re-run `python src/evaluate.py --split test --weights ...` and show generated metrics, explaining the tiny illustrated split and lack of real photo validation.  
**4:45–5:00** Describe remaining work and limitations. 

Screen capture the actual application—not slides alone. Keep the weights, test inputs and video saved offline.
