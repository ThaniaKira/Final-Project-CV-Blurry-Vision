import os
import cv2
import glob
import matplotlib.pyplot as plt

def generate_contact_sheet(sample_dir="data/samples", out_path="outputs/contact_sheet.png"):
    images = sorted(glob.glob(f"{sample_dir}/*.png"))[:20]
    if not images:
        print("No sample images found.")
        return
        
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    fig.suptitle("Dataset Contact Sheet (20 Labeled Gesture Samples)", fontsize=16, fontweight="bold")

    for idx, path in enumerate(images):
        ax = axes[idx // 5, idx % 5]
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = os.path.basename(path)
        
        ax.imshow(img_rgb)
        ax.set_title(f"#{idx+1:02d}: {label[:15]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Contact sheet saved to '{out_path}'.")

if __name__ == "__main__":
    generate_contact_sheet()
