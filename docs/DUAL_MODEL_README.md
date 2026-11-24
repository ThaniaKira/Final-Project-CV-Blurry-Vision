# Blurry Vision · Dual-backend package (original UI preserved)

## Status, first

**No HaGRID-trained weights are supplied in this ZIP.** The environment used to prepare the backend had no network route to the public HaGRID images; the uploaded ZIP itself contained only 20 illustrated core samples and `models/gesture_contour.npz`. In particular, **100% on six illustrations is not 100% accuracy on real hands**. Do not claim that HaGRID-trained models or their accuracy have been validated until the training script has run and its evaluation JSON is present.

The original `web/` HTML, CSS, JavaScript, images and page routes are byte-for-byte unchanged. The original `models/gesture_contour.npz` remains as a historical six-class illustrative baseline.

## Two independent model heads

| Head | Weight file expected | Training labels | Website behavior without weights |
|---|---|---|---|
| Core six | `models/gesture_hagrid_mediapipe.npz` | Open_Hand, Fist, Peace, Thumbs_Up, Pointing, OK | Existing untrained MediaPipe geometry rules; original Studio output remains six-gesture-first |
| Numbers | `models/numbers_hagrid_mediapipe.npz` | Number_0 through Number_5 | Untrained MediaPipe finger geometry heuristics (zero calibrated confidence) |

The new `src/dual_models.py` shares a **single MediaPipe detector** between the two heads to avoid roughly doubling webcam inference latency. Independent Random Forest / ExtraTrees classifiers can be loaded from the two files without interfering with each other.

### API (no front-end changes)

- `POST /api/analyze` (unchanged contract): existing core-gesture annotated PNG and `details.gesture`; also adds `details.number_prediction` for later game integration.
- `POST /api/analyze/numbers` (new): the same uploaded photo returns `Number_0` .. `Number_5` or `Not a number gesture`, drawn into the response image.
- `GET /api/status`: original keys remain; adds `number_model` status including whether weights exist.

The numeric fallback deliberately rejects some non-numeric gestures (OK, rock, thumbs up). It is a **prototype heuristic**, not the trained second model.

## Train real HaGRID models remotely, WITHOUT extracting the images onto your PC

Use Kaggle or Colab, where the approximately **1 GB** public [HaGRID 30K 384p sample](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p) ZIP is kept on cloud storage; do **not** put it in your project Git repository.

In a **Python 3.11** cloud environment:

1. Upload this backend project (or clone your repo plus these changes) to cloud compute.
2. Install `requirements.txt` in the cloud runtime. The pinned MediaPipe v0.10.14 uses the `solutions.hands` API expected by this project.
3. Download/attach `hagrid-sample-30k-384p.zip` to the **cloud runtime** (not your laptop), either through Kaggle Dataset attachment or `huggingface_hub.hf_hub_download(repo_id="cj-mills/hagrid-sample-30k-384p", filename="hagrid-sample-30k-384p.zip", repo_type="dataset")`.
4. Run:

```bash
python src/train_dual_hagrid.py --archive /path/in/cloud/hagrid-sample-30k-384p.zip --max-per-class 250
```

This performs *subject-disjoint* train/validation/test splitting using HaGRID `user_id` and reads individual image bytes from within the ZIP. It extracts real MediaPipe landmarks, trains both heads, chooses Random Forest or ExtraTrees by validation macro-F1, evaluates once on held-out test images, and writes:

```text
models/gesture_hagrid_mediapipe.npz
models/gesture_hagrid_mediapipe.evaluation.json
models/numbers_hagrid_mediapipe.npz
models/numbers_hagrid_mediapipe.evaluation.json
models/dual_models_report.json
```

Download only these small output files from the cloud into your local `models/` directory. Restart the website (`python src/app.py`) to load the trained weights. No HaGRID JPG collection should be stored locally.

**Note:** HaGRID `one` is used as a proxy for the existing `Pointing` UI class; validate that its pose matches your educational activity. In numeric mode, `peace`/`two_up` both map to 2, and `three`/`three2` both map to 3. Each head is evaluated independently. User-independent splitting avoids leakage between the two heads; source labels still need human verification for children's specific counting style.

## Evaluate and explain your claims

- Keep the original six-illustration baseline as a **separate** result: on the six *supplied illustrated test images*, the existing contour weight file predicted 6/6 correctly. This is not a HaGRID test or evidence of real-world generalization.
- Report `test.accuracy_all_images` and `test.macro_f1_all_images` from each generated `.evaluation.json`; failed hand detections count against accuracy.
- Perform a **separate** controlled webcam evaluation with children or adult volunteers (consent and privacy safeguards as applicable), differing skin tones, backgrounds, camera distances, and lighting. No numeric or core accuracy number can be guaranteed in advance.
- Run `pytest -q` for code-level checks, and verify the full desktop UI and webcam manually on a machine with the pinned MediaPipe dependencies. Tests aren't a substitute for a real dataset benchmark.

## Child-friendly game direction (future UI extension)

For the smallest educational feature, add an optional `/play` route *in a future UI change*: show one target number (e.g., `3`), have the child hold up that many fingers, call `/api/analyze/numbers`, and award a point only after the same confident number is detected for several frames. Avoid a strict countdown and support keyboard/tap controls so every child can participate. **No `/play` page has been added** because the request was to preserve the original website UI.
