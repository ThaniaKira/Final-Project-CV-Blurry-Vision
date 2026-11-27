# Blurry Vision v1.3 — Child-Friendly Live Games

## What's new

1. **Count & Shine** (`/play?game=count`): mixed order of numbers 1–5; large number and its English word are shown and spoken. Five rounds.
2. **Copy the Pose** (`/play?game=match`): references all six core poses with illustration cards, classified by the existing six-class model. Six rounds.
3. **Finger Math** (`/play?game=math`): simple sums/differences whose answers are always between one and five, classified by the existing number model. Five rounds.
4. **Memory Trail** (`/play?game=memory`): shows, speaks, and hides 2-, 3- and 4-step number sequences. Three rounds.

Each game checks actual webcam predictions with `POST /api/live?mode=numbers` or `POST /api/live?mode=core`. Each positive answer requires two matching frames above the relevant vote-score threshold. The child must change or lower their hand between repeated equal answers. No fake hand labels or arbitrary correctness are used. Lowering the hand or changing its pose re-arms the next challenge.

## Real-time behavior

- The browser preview remains live and the server is queried automatically on a self-scheduled loop. No overlapping requests are launched by each page. Frames are reduced to a maximum width of approximately 520px for games and 480px for Studio, JPEG quality 0.76–0.78, to minimize latency.
- `POST /api/live` returns only predictions and landmarks, not a base64-rendered image. An independent `stream=True` MediaPipe tracker is reused for live frames; standard `/api/analyze` and `/api/analyze/numbers` retain their original one-shot image behavior and responses.
- The Studio draws bounding boxes and hand keypoints over the live preview in a canvas. Upload mode remains available for offline images.
- Both trained `.npz` models are unchanged. This release does *not* train a new model, alter class mappings, or recalculate historical metrics.

## Sound

- `/static/audio/star_steps_original.wav` is an originally synthesized instrumental composition created for this project. The **Music On/Off** control is accessible on all pages. Audio playback starts only following a user click because browsers typically restrict autoplay. The soundtrack pauses on hidden tabs.
- Browser Speech Synthesis speaks learning instructions, including "one". Speech can be replayed with **Hear it again**. On the Gesture Guide, number buttons speak numbers 1–5.
- English voices vary by browser/OS; speech may be unavailable if the browser lacks Speech Synthesis or appropriate voices. The visual labels remain.

## Usability and safety

- Children should use the webcam with an adult's guidance. The game requests camera permission explicitly and stops camera tracks when leaving the page.
- Webcam images are sent to the site's **local** Python service at 127.0.0.1; no image data are saved by these endpoints, nor sent to external ML APIs. Do not expose the development server to the public internet.
- The game is designed for one hand; it does not count two hands to produce numbers above five.

## Evaluate before retraining

Record at least 20–30 live trials per class across different people, lighting conditions and camera distances. Record (a) MediaPipe hand detection rate and (b) gesture classification accuracy **conditional on successful hand detection**, and the complete end-to-end game success rate. Keep the test people separate from the training people.

A `No hand detected` status means the classifier did not receive valid hand features. Improve light, distance, framing and detector settings before investing in more classifier training. If detection succeeds but gesture labels are frequently incorrect, use more correctly labeled diverse HaGRID/custom data and validate on held-out individuals. Children's poses and small hands should be represented in consented, privacy-respecting evaluation data. Accuracy on the adult-centric public HaGRID images may not transfer to children.

## Developer quick checks

```powershell
.\.venv\Scripts\python.exe src/app.py
# Visit http://127.0.0.1:7860/play
# Visit http://127.0.0.1:7860/studio, choose Use camera
```

The browser and server must be running on the **same computer** for localhost webcam capture. Only one camera tab is recommended at a time. Self-test both trained model paths under `/api/status` and `/models`.

## Git versioning

This update follows v1.2; consider a **new** feature commit and `v1.3` tag once you have tested the updated folder. Do not move v1.0 or v1.2 tags.
