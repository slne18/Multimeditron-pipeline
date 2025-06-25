import json
import random
import math
from tqdm import tqdm
from collections import defaultdict
import os

radiopedia_path = "/mloscratch/users/noize/IRM_jsonl/IRM_radiopedia.jsonl"
medtrinity_path = "/mloscratch/users/noize/IRM_jsonl/MRI_med.jsonl"
combined_output = "/mloscratch/users/noize/IRM_jsonl/MRI_combined.jsonl"

# Load JSONL
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save JSONL
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Saving... {os.path.basename(path)}"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Group Radiopedia data by patient
def group_radiopedia_by_patient(records):
    patient_groups = defaultdict(list)
    for rec in tqdm(records, desc="Grouping Radiopedia by patient"):
        path = rec["modalities"][0]["value"]
        parts = os.path.normpath(path).split(os.sep)
        patient_id = parts[-3] if len(parts) >= 3 else parts[-2]
        patient_groups[patient_id].append(rec)
    return list(patient_groups.values())

if __name__ == "__main__":
    print("Loading files...")
    radiopedia_data = load_jsonl(radiopedia_path)
    medtrinity_data = load_jsonl(medtrinity_path)

    # Grouping
    radiopedia_groups = group_radiopedia_by_patient(radiopedia_data)
    medtrinity_groups = [[item] for item in medtrinity_data]

    # Combine all
    all_groups = radiopedia_groups + medtrinity_groups
    random.shuffle(all_groups)

    # Save combined dataset before splitting
    combined_dataset = [item for group in all_groups for item in group]
    save_jsonl(combined_dataset, combined_output)
    print(f"Combined dataset saved to: {combined_output}")
