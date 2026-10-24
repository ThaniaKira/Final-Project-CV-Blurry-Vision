import os
import glob
import json
import cv2
import pandas as pd

GESTURE_CLASSES = ["Open_Hand", "Fist", "Peace", "Thumbs_Up", "Pointing", "OK"]

class GestureDataLoader:
    """Loads, indexes, and splits HaGRID sample gesture dataset."""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.samples_dir = os.path.join(data_dir, "samples")
        self.annotation_path = os.path.join(data_dir, "hagrid_sample_annotations.json")
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, "r") as f:
                return json.load(f)
        return {}

    def get_sample_images(self):
        """Finds all PNG/JPG sample images in the samples directory."""
        patterns = [os.path.join(self.samples_dir, "*.png"), os.path.join(self.samples_dir, "*.jpg")]
        images = []
        for p in patterns:
            images.extend(glob.glob(p))
        return sorted(images)

    def load_splits(self):
        """Loads train, val, and test split images and labels."""
        splits = {"train": [], "val": [], "test": []}
        
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                continue
            for cls_name in GESTURE_CLASSES:
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.exists(cls_dir):
                    continue
                for img_file in os.listdir(cls_dir):
                    if img_file.endswith((".png", ".jpg")):
                        img_path = os.path.join(cls_dir, img_file)
                        splits[split].append((img_path, cls_name))
        return splits["train"], splits["val"], splits["test"]
