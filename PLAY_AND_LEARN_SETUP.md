# Blurry Vision — Play & Learn + detection fallback

## What was changed

- NEW `http://127.0.0.1:7860/play`: number-counting game (1–5), webcam and photo inputs, score, five rounds, optional periodic checks.
- Navigation links to Play & Learn; existing Home, Studio, Guide content and styling preserved.
- Existing trained model weights are **byte-for-byte unchanged**: `models/gesture_hagrid_mediapipe.npz` and `models/numbers_hagrid_mediapipe.npz`.
- `/api/analyze/numbers` is used by the game. `Number_1` through `Number_5` map to the game actions.
- MediaPipe detector keeps the normal camera pass, then contrast enhancement, then a padded/contrast fallback when earlier passes fail. Landmark coordinates from padded images are mapped back to original image pixels before classification.
- Studio now displays N/A rather than an inappropriate 0% classifier score when no hand is detected.
- API `details.detection_attempt` can be `original`, `contrast`, `padded_contrast`, or `none`. Useful for debugging.

## How to run (Windows / VS Code)

1. Extract this ZIP and open the `Blurry_Vision_Phase5_UI_UX` folder in VS Code.
2. Use **Python 3.11**, preferably in a new project virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements_website.txt
   .\.venv\Scripts\python.exe src/app.py
   ```
3. Open `http://127.0.0.1:7860/play` in Chrome or Edge and allow camera permission.
4. For the existing Studio visit `/studio`.

If you already have the previous ZIP running, stop its server (`Ctrl+C`), open this **new** folder, and start `src/app.py` again. Check `/api/status` for both weights present.

## Why this does not guarantee improved accuracy

Your user screenshot shows a closed fist close to the lens with uneven room light. A missed hand means MediaPipe did not return landmarks, **before the ExtraTrees gesture classifier could run**. More training photos for the ExtraTrees model won't directly fix missed palm detections. The fallback may help some pictures but can increase latency and needs real webcam testing.

Improve camera placement first: move roughly an arm's length away, show the whole hand including wrist, place a light in front of the palm, and try palm/open-hand then fist. Retest across people and lighting. Some webcam settings may also cause blur/exposure issues.

The original HaGRID evaluation was 89.35% all-image accuracy (core) and 88.60% (numbers) in the supplied report. These scores are **not** accuracy measurements for this updated detector or for children using the game. The game has a 0.55 *vote-fraction* guard and requires 2 consecutive matching predictions only in auto mode; these are UX heuristics and are not validated thresholds.

## Recommended real-world test

For each child/person and gesture, take at least 10 captures in bright, moderate, and dim lighting. Record both (a) hand detected? and (b) correct gesture given detection? separately. Include hard cases. Keep all video frames of the same person in a single train/validation/test split if you later train a new model. Parental permission and privacy considerations apply for images involving children.
