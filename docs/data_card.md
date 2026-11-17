# Dataset card — archive-derived checkpoint 4

## Source and verified inventory
The original uploaded ZIP contains **20 illustrated (synthetically drawn) 224 × 224 pixel PNGs** labeled as six gestures: Open_Hand, Fist, Peace, Thumbs_Up, Pointing, OK. These drawings, not actual hand photographs, are the source of the current training and evaluation subset. There are also 20 unrelated JPGs named `call`, `dislike`, `like`, and `mute` under `data/samples`; they are not part of this six-class experiment. The repository includes a file named `hagrid_sample_annotations.json`, but its uniform bounding boxes are not evidence of HaGRID images or authentic reviewed annotations. A HaGRID licence/derivation has **not** been verified from the ZIP.

| Split | Number | Per-class coverage |
|---|---:|---|
| `data/train` | 8 | Open_Hand 2, Fist 2, other classes 1 each |
| `data/val` | 6 | 1 each |
| `data/test` | 6 | 1 each |
| **Total** | **20** | 6 classes |

This is **40%/30%/30%**, not the 70%/15%/15% stated in some historical reports. Images may originate from the same drawing generator; **source/person disjointness cannot be established**, and representativeness for real webcam users is absent. Evaluation metrics on six images have extremely wide uncertainty, and cannot validate 80% classroom lighting robustness.

## Labeling, preprocessing, and usage
The class is encoded in the filename and folder name. The colored dots visible on the drawings are part of the provided sample artwork, not verified MediaPipe landmarks. The inference pipeline resizes to 224 × 224, applies CLAHE enhancement and YCrCb skin thresholding (contour model), then extracts a region-derived silhouette grid, Hu moments and geometric features. The alternate MediaPipe backend extracts real 21 × 3 coordinates but must be trained on real labeled hand photographs before use.

## Data collection needed before deployment
Collect opt-in, consented webcam photographs/videos from multiple people and environments (varied lighting, skin tones, backgrounds, camera angles, hand sizes and motion). Organize by **participant/video group** into fixed train/validation/test groups. Write source/licence/consent records, counts per class, and annotation review protocol. Do not add frame-near-duplicates across splits. Re-evaluate on a genuinely unseen set before claiming the original targets.
