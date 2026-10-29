# Data Card: HaGRID Educational Gesture Dataset Sub-Set

## 1. Overview & Source
- **Dataset Name**: HaGRID (HAnd GESTURE RECOGNITION IMAGE DATASET) & Custom Webcam Captures
- **Primary Source**: HaGRID Public Dataset (Kaggle / GitHub) & Local Preschool Classroom Captures
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Target Task**: Multi-class static hand gesture classification for early childhood touchless interaction.

## 2. Gesture Classes & Counts
Target 6 Gesture Classes:
- `Open_Hand` (5 / Open Palm)
- `Fist` (0 / Closed Fist)
- `Peace` (2 / V-sign)
- `Thumbs_Up` (Affirmative gesture)
- `Pointing` (1 / Index finger)
- `OK` (Circle gesture)

### Data Splits (By Subject Source):
- **Total Samples**: 20 attached representative gesture PNG images (plus expanded Kaggle set)
- **Training Set (70%)**: 8 images across subject splits (`data/train/`)
- **Validation Set (15%)**: 6 images across subject splits (`data/val/`)
- **Testing Set (15%)**: 6 images locked for evaluation (`data/test/`)

## 3. Labeling Method & Annotations
- **Labeling Standard**: Manual verification & bounding box coordinate normalization $[x_{min}, y_{min}, w, h]$.
- **Bounding Boxes**: Encoded in `data/hagrid_sample_annotations.json`.

## 4. Preprocessing Pipeline (`src/preprocess.py`)
1. **Spatial Normalization**: Resized to $224 \times 224$ pixels.
2. **Noise Reduction**: $5 \times 5$ Gaussian blur filter.
3. **Skin Masking**: Conversion to YCbCr color space and adaptive thresholding ($Cr \in [133, 173]$, $Cb \in [77, 127]$).
4. **Blur Estimation**: Laplacian variance calculation to flag motion or defocus blur ($> 100$ is sharp).
