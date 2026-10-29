import cv2
import numpy as np

def load_and_resize(image_path_or_array, target_size=(224, 224)):
    """Loads image from path or accepts ndarray, then resizes to target_size."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path_or_array}")
    else:
        img = image_path_or_array.copy()
    return cv2.resize(img, target_size)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """Applies Gaussian Blur for noise reduction and smoothing."""
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_clahe(image):
    """Enhances image contrast using CLAHE on the L channel in LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def extract_ycbar_skin_mask(image):
    """Extracts skin color region mask in YCrCb color space."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Standard human skin tone thresholds in YCrCb space
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def compute_blur_score(image):
    """Calculates Laplacian variance as a quantitative metric for image motion blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def preprocess_pipeline(image_path_or_array, target_size=(224, 224), use_clahe=True):
    """Complete shared preprocessing pipeline."""
    resized = load_and_resize(image_path_or_array, target_size)
    if use_clahe:
        resized = apply_clahe(resized)
    denoised = apply_gaussian_blur(resized)
    skin_mask = extract_ycbar_skin_mask(denoised)
    segmented = cv2.bitwise_and(denoised, denoised, mask=skin_mask)
    blur_score = compute_blur_score(denoised)
    return resized, skin_mask, segmented, blur_score
