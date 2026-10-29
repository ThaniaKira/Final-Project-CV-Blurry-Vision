import os
import glob
import cv2
import numpy as np

class GestureDataLoader:
    """Dataset loader supporting 70/15/15 train/val/test splits."""
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.classes = ["Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK"]
        
    def load_split(self, split="train"):
        """Loads all images and class labels for a given split ('train', 'val', 'test')."""
        split_dir = os.path.join(self.data_root, split)
        images = []
        labels = []
        filepaths = []
        
        if not os.path.exists(split_dir):
            # Fallback to loading all from data/samples/
            samples_dir = os.path.join(self.data_root, "samples")
            for img_p in sorted(glob.glob(os.path.join(samples_dir, "*.png"))):
                fname = os.path.basename(img_p)
                for c in self.classes:
                    if c.lower() in fname.lower():
                        img = cv2.imread(img_p)
                        if img is not None:
                            images.append(img)
                            labels.append(c)
                            filepaths.append(img_p)
                        break
            return images, labels, filepaths

        for c in self.classes:
            cdir = os.path.join(split_dir, c)
            if not os.path.exists(cdir):
                continue
            for img_p in sorted(glob.glob(os.path.join(cdir, "*.png")) + glob.glob(os.path.join(cdir, "*.jpg"))):
                img = cv2.imread(img_p)
                if img is not None:
                    images.append(img)
                    labels.append(c)
                    filepaths.append(img_p)
                    
        return images, labels, filepaths
