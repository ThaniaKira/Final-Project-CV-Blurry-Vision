# Blurry Vision v1.4 — Child-Friendly Game Improvements

This update builds on v1.3 and leaves the original website theme, backend endpoints, and trained model weights unchanged.

## Requested changes

1. **Count & Shine** still shuffles the five numbers each new game. Each number is displayed once per five-round session.
2. **Copy the Pose** now displays six separate illustrated hand-pose PNG assets from `web/static/img/pose-guides/`. They are not crops from the recognition output or model test samples. A confidently misrecognized pose (model vote confidence >= 0.50) triggers a gentle correction naming the expected pose, e.g., “Good try! Peace sign gesture, please.” The verbal correction has a cooldown so it won't repeat every camera frame.
3. **Finger Math** uses encouraging, more varied English prompts and gently says the intended finger count when a wrong number is confidently recognized.
4. **Memory Trail** reads the complete number pattern **twice** before it is hidden. It does this on initial reveal and when the child chooses “Hear Sequence Twice Again.” The 75-second answer timer starts *after* the pattern presentation.
5. **Gentle countdowns:** Finger Math has 60 seconds per problem. Memory Trail has 75 seconds per pattern. The clock pauses when the browser tab is hidden and only starts once the camera is running; if time runs out, the child can try again without losing any stars. Count & Shine and Copy the Pose are untimed so beginners can practice without pressure.
6. **Voice selector:** Parents/teachers can pick from English voices installed on the local device. The game tries to choose a natural-sounding voice automatically, lowers background music during speech, and uses warmer phrasing. Speech quality ultimately depends on the operating system and browser; the package does not contain or claim to contain a human-recorded voice.

## Running the updated project (Windows / VS Code)

From your VS Code project root, with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_website.txt
.\.venv\Scripts\python.exe src/app.py
```

Then visit `http://127.0.0.1:7860/play` and choose a game. Grant camera permission. The games require the local Python server for actual predictions; the game's results are not simulated.

## Model files

Existing files `models/gesture_hagrid_mediapipe.npz` and `models/numbers_hagrid_mediapipe.npz` are unchanged. No additional training was performed, and no new accuracy figures are claimed. Run real-life tests with children under adult supervision before classroom use.

## Applying update-only ZIP to a Git working folder

Extract the archive, then copy its `Blurry_Vision_Phase5_UI_UX/` contents into your cloned Git working folder. Preserve your local `.git`, `.venv`, and any custom changes. Review `git status` before staging or committing.
