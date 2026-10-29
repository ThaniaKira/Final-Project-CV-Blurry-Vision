import os
import kagglehub

def download_hagrid_samples():
    """Downloads sample subset of HaGRID dataset from Kaggle Hub."""
    print("Downloading HaGRID sample dataset...")
    try:
        path = kagglehub.dataset_download("hpc-ai/hagrid")
        print(f"HaGRID downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"KaggleHub download notice: Using local data/samples subset. Details: {e}")
        return "./data/samples"

if __name__ == "__main__":
    download_hagrid_samples()
