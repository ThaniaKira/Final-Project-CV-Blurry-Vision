import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class GestureDataLoader:
    def __init__(self, data_dir="data/samples", img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ["Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK"]
        
    def load_dataset(self):
        images = []
        labels = []
        filepaths = sorted(glob.glob(os.path.join(self.data_dir, "*.png")))
        
        for path in filepaths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            
            label_str = os.path.basename(path).split("_")[-1].replace(".png", "")
            if label_str in self.classes:
                labels.append(self.classes.index(label_str))
            else:
                labels.append(0)
            images.append(img)
            
        return np.array(images), np.array(labels)

    def split_data(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        X, y = self.load_dataset()
        if len(X) == 0:
            print("No local sample images found.")
            return None, None, None
            
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_ratio + test_ratio), random_state=42)
        relative_val = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_val), random_state=42)
        
        print(f"Data Split Complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    loader = GestureDataLoader()
    loader.split_data()
