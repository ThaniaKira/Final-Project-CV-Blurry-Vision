import os
import glob
import cv2
import numpy as np

def draw_prediction_overlay(image, label, confidence, blur_score, hand_landmarks=None):
    """Renders annotated bounding box, landmark skeleton, confidence %, and blur score on image."""
    annotated = image.copy()
    h, w, _ = annotated.shape
    
    # Header banner
    cv2.rectangle(annotated, (0, 0), (w, 45), (20, 20, 20), -1)
    
    # Text overlay
    header_text = f"Gesture: {label} ({confidence*100:.1f}%) | Blur: {blur_score:.1f}"
    cv2.putText(annotated, header_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    
    # Bounding Box
    box_color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
    margin = 30
    cv2.rectangle(annotated, (margin, margin+40), (w-margin, h-margin), box_color, 2)
    
    # Render MediaPipe landmarks if available
    if hand_landmarks is not None:
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(annotated, (cx, cy), 2, (255, 255, 255), -1)
            
    return annotated

def generate_contact_sheet(image_paths, output_path="data/contact_sheet.png", grid_size=(4, 5)):
    """Generates a 20-sample labeled grid contact sheet."""
    rows, cols = grid_size
    cell_w, cell_h = 160, 160
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, path in enumerate(image_paths[:rows*cols]):
        r, c = idx // cols, idx % cols
        img = cv2.imread(path)
        if img is None:
            continue
        resized = cv2.resize(img, (cell_w, cell_h))
        
        fname = os.path.basename(path)
        label = fname.split("_")[2] if len(fname.split("_")) > 2 else "Gesture"
        
        cv2.putText(resized, label, (5, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        sheet[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
        
    cv2.imwrite(output_path, sheet)
    return output_path
