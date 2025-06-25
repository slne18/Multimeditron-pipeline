import json
import random
from tqdm import tqdm

input_path = "/mloscratch/users/noize/IRM_jsonl/MRI_combined_clean.jsonl"  
output_path = "/mloscratch/users/noize/IRM_jsonl/MRI_combined_clean_1M.jsonl"
num_samples = 1000000  # Number of samples to keep

# Load JSONL
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save JSONL
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Saving... {os.path.basename(path)}"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Load the large dataset
    print("Loading dataset...")
    dataset = load_jsonl(input_path)

    # Print the initial number of samples
    print(f"Initial number of samples: {len(dataset)}")

    # Randomly shuffle the dataset
    random.shuffle(dataset)

    # Keep only 500k samples
    sampled_dataset = dataset[:num_samples]

    print(f" Sampled dataset contains {len(sampled_dataset)} examples.")

    # Save the sampled dataset
    save_jsonl(sampled_dataset, output_path)
