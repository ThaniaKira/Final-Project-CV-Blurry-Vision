import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.preprocess import preprocess_pipeline
from src.data_loader import GestureDataLoader

def extract_color_moments(image):
    """Extracts mean, std, and spatial color moments from YCbCr skin region."""
    _, mask, segmented, _ = preprocess_pipeline(image, target_size=(128, 128))
    ycrcb = cv2.cvtColor(segmented, cv2.COLOR_BGR2YCrCb)
    
    # Calculate Mean and Std for Y, Cr, Cb
    mean_val, std_val = cv2.meanStdDev(ycrcb, mask=mask)
    
    # Spatial grid distribution of skin mask
    resized_mask = cv2.resize(mask, (16, 16)).flatten() / 255.0
    
    features = np.hstack([mean_val.flatten(), std_val.flatten(), resized_mask])
    return features

class ClassicalBaselineModel:
    """Classical YCbCr Skin Color Mask + Spatial Moment + kNN Baseline."""
    def __init__(self, n_neighbors=1):
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False
        
    def fit(self, X_images, y_labels):
        X_feats = [extract_color_moments(img) for img in X_images]
        self.clf.fit(X_feats, y_labels)
        self.is_trained = True
        
    def predict(self, X_images):
        X_feats = [extract_color_moments(img) for img in X_images]
        return self.clf.predict(X_feats)

def run_baseline():
    """Trains and evaluates classical baseline model on train/val splits."""
    loader = GestureDataLoader()
    train_imgs, train_lbls, _ = loader.load_split("train")
    val_imgs, val_lbls, _ = loader.load_split("val")
    
    if len(train_imgs) == 0:
        train_imgs, train_lbls, _ = loader.load_split("samples")
        val_imgs, val_lbls = train_imgs[:6], train_lbls[:6]
        
    model = ClassicalBaselineModel()
    model.fit(train_imgs, train_lbls)
    preds = model.predict(val_imgs)
    
    acc = accuracy_score(val_lbls, preds)
    macro_f1 = f1_score(val_lbls, preds, average="macro", zero_division=0)
    
    print("==================================================")
    print("Classical Baseline Evaluation (Skin Color + kNN)")
    print("==================================================")
    print(f"Validation Accuracy : {acc*100:.2f}%")
    print(f"Validation Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_lbls, preds, zero_division=0))
    return acc, macro_f1

if __name__ == "__main__":
    run_baseline()
