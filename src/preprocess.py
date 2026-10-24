import cv2
import numpy as np

def load_and_resize(image_path_or_array, target_size=(224, 224)):
    """Loads image from path or array and resizes to target spatial resolution."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array.copy()
    else:
        raise ValueError("Invalid image input type. Expected path or numpy array.")
    return cv2.resize(img, target_size)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur filter to reduce high-frequency image noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def extract_skin_mask(image):
    """Converts BGR image to YCbCr color space and isolates skin regions using adaptive thresholding."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Typical YCbCr skin range: Y in [0, 255], Cr in [133, 173], Cb in [77, 127]
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Morphological opening and closing to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def compute_blur_score(image):
    """Computes Laplacian variance to estimate image sharpness vs motion/defocus blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def extract_hand_contour(mask):
    """Extracts the largest external contour representing the main hand object."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def preprocess_pipeline(image_path_or_array, target_size=(224, 224)):
    """Executes full preprocessing pipeline: Resize -> Gaussian Blur -> YCbCr Skin Mask -> Blur Score."""
    resized = load_and_resize(image_path_or_array, target_size)
    blurred = apply_gaussian_blur(resized)
    mask = extract_skin_mask(blurred)
    segmented = cv2.bitwise_and(resized, resized, mask=mask)
    blur_score = compute_blur_score(resized)
    return resized, mask, segmented, blur_score
