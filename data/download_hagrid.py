import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_hagrid_dataset(file_path=""):
    print("Fetching HaGRID dataset via KaggleHub...")
    try:
        hf_dataset = kagglehub.load_dataset(
            KaggleDatasetAdapter.HUGGING_FACE,
            "kapitanov/hagrid",
            file_path,
        )
        print("Successfully loaded HaGRID Dataset:")
        print(hf_dataset)
        return hf_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    load_hagrid_dataset()
