# Phase 4 upgrade — HaGRID Kaggle real-hand training (six gestures)

This upgrade addresses the **Open Hand mistakenly labeled Pointing** webcam failure: the legacy skin contour detector selected part of the student's body, not the hand. For real-hand photos, use MediaPipe's 21 real hand landmarks. The pretrained MediaPipe **hand detector** finds the hand, and the Random Forest trained on Kaggle HaGRID data assigns one of the six gestures.

## Source and six-class label mapping

Kaggle: <https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p> (HaGRID Sample 30k, downscaled to 384p; approximately 1 GB download). Original dataset: <https://www.kaggle.com/datasets/kapitanov/hagrid>. The six source labels are:

| HaGRID label | Blurry Vision label | Action |
|---|---|---|
| `palm` | `Open_Hand` | High-five / open hand |
| `fist` | `Fist` | Closed hand |
| `peace` | `Peace` | Two-finger V |
| `like` | `Thumbs_Up` | Thumb up |
| `one` | `Pointing` | One extended index finger (typically pointing up) |
| `ok` | `OK` | Finger/thumb OK sign |

The sampled distribution, subject identities, and recording conditions affect the actual accuracy. `Pointing` specifically reflects the dataset's `one` gesture; pointing sideways may behave differently. The original HaGRID samples are adult hands and may not generalize to preschool children's hands.

## Windows / Visual Studio Code — exact commands

Use a **new VS Code terminal** at `Blurry_Vision_CP4`, with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell refuses the activation script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry.

**1. Download the HaGRID sample from Kaggle (requires internet).** The easy command is:

```powershell
python -m pip install kaggle
python src/prepare_kaggle.py --download --max-per-class 400
```

The downloader calls the official `kaggle` command. Some Kaggle configurations may require signing in or adding credentials; follow Kaggle's instructions if you receive a 401/403 error. Never commit your private Kaggle credentials.

**Alternative if direct download fails:** Open <https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p>, download its ZIP manually, then run:

```powershell
python src/prepare_kaggle.py --archive "D:\Downloads\hagrid-sample-30k-384p.zip" --max-per-class 400
```

Or, if you've extracted the Kaggle ZIP, use `--source-dir "D:\path\to\hagrid-sample-30k-384p"`. The importer reads real JPEGs and JSON annotations directly from the archive, crops the hand with a margin, and writes images in `data/hagrid_real/{train,val,test}/{class}`. It does **not** mix the original illustrated images into the real-hand dataset.

The importer splits by HaGRID **`user_id`**, globally across the six classes. A person photographed for training can never appear in validation or test. It creates `data/hagrid_real/manifest.csv` and `data_card.json`. The default size cap is **up to 400 annotated photos/class**, not 400 guaranteed successful MediaPipe hand detections. If you need more photos, prepare to a fresh location using `--max-per-class 0` for all matched images, or 600 etc.

**2. Train the actual MediaPipe landmark + Random Forest model.**

```powershell
python src/train.py --backend mediapipe --data data/hagrid_real --output models/gesture_hagrid_mediapipe.npz
```

This can run on a CPU laptop. It requires enough real hand photos for MediaPipe to find hand landmarks; it reports detection coverage on validation. It compares three Random Forest settings using validation macro-F1, chooses one, and saves a portable NPZ. Training-only horizontal mirroring is enabled. The **test set is not used** to select or fit the model.

**3. Evaluate your REAL held-out test images.**

```powershell
python src/evaluate.py --split test --data data/hagrid_real --weights models/gesture_hagrid_mediapipe.npz --output outputs/hagrid_eval
```

Read the generated `outputs/hagrid_eval/metrics.json`, `predictions.csv`, and `confusion_matrix.png`. Accuracy, macro-F1, per-class F1, missed hand detections, and warm-start latency are measured on **your actual computer**. The test script audits that no HaGRID user_id overlaps across train/validation/test. Repeatedly changing the model based on test outcomes would invalidate the held-out evaluation; make changes against validation instead.

**4. Open the trained model in the main Phase 4 web app.**

```powershell
python src/app.py
```

Open <http://127.0.0.1:7860>. Choose **Kaggle HaGRID trained: MediaPipe + Random Forest** from the method dropdown. If your trained model doesn't appear, close and restart the app after training; the app scans the models folder at launch.

For live video:

```powershell
python src/demo_stream.py --webcam 0 --mode kaggle
```

Use `--mode rules` for the instant rule-based real-hand demo (works without Kaggle training), or `--mode legacy` to reproduce the illustrated contour model's limitations.

## What an 80–90% score means

- **80–90% is an objective, not an output hardcoded in the program.** It can be higher or lower after real training and evaluation.
- Report the measured *held-out Kaggle* accuracy and macro-F1, plus no-hand detection percentage, and class-level failures. Keep the older **6 illustrated test images / 100%** results separate; those do not establish real-world accuracy.
- The HaGRID 30K sample has numerous hand poses photographed under many conditions, but a subject-disjoint 6-class assessment on this subset is a **new experiment**, not a score published by Kaggle.
- Test live webcam cases separately (at least ten, including harsh lighting, clutter, occlusion). Take real photos with informed consent before including children. Do not claim real-time performance without timing it on the demonstration laptop.
- **If held-out accuracy is below 80%:** inspect the confusion matrix, increase per-class training photos, fix low detection coverage, test the dataset's hand crops, and retune using validation only. Do not manipulate the test set to hit a target.

## If the app shows the wrong label

Take a clear image in indoor daylight with a full hand in frame. Verify the landmark points actually align with the fingers. If `No hand detected`, move the hand closer to the camera. For prediction errors with correct landmarks, collect more examples of the confused classes and compare their class-level recall. `one` can be confused with some rotated `like` poses depending on camera viewpoint.

## Academic reproducibility notes

The report must state: dataset source/link and license (HaGRID sample lists CC BY-SA 4.0), downloaded source version/date, selected per-class counts, user_id-based split, seed 42, MediaPipe/scikit-learn versions, hyperparameters, final NPZ hash, performance metrics, failure examples, hardware details and demo laptop speed.

**The ZIP does NOT contain the 1 GB Kaggle dataset or a Kaggle-trained model.** Both are generated locally using the commands above; **until then, the built-in real-hand demo is geometry rules, not trained Kaggle Random Forest**. Do not submit illustrative accuracy figures as new results.
