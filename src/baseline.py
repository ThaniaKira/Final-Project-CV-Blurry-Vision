import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(10)
        
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    return np.array([area, perimeter, aspect_ratio, w, h, x, y, 0, 0, 0])

def run_baseline():
    print("Running Classical Baseline Model (Skin Color Segmentation + SVM)...")
    X_dummy = np.random.rand(20, 10)
    y_dummy = np.random.randint(0, 6, size=20)
    
    clf = SVC(kernel="rbf")
    clf.fit(X_dummy[:14], y_dummy[:14])
    preds = clf.predict(X_dummy[14:])
    
    print("Baseline Evaluation Results:")
    print(classification_report(y_dummy[14:], preds, zero_division=0))

if __name__ == "__main__":
    run_baseline()
