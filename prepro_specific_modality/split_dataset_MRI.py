import json
import random
from tqdm import tqdm
from collections import defaultdict
import os

combined_input = "/mloscratch/users/noize/IRM_jsonl/MRI_combined_clean_1M.jsonl"
train_output = "/mloscratch/users/noize/IRM_jsonl/IRM-train.jsonl"
test_output = "/mloscratch/users/noize/IRM_jsonl/IRM-test.jsonl"
train_ratio = 0.8

# Load flat JSONL (1 line = 1 sample)
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save flat JSONL
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Saving {os.path.basename(path)}"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Group samples by patient
def group_by_patient(records):
    patient_groups = defaultdict(list)
    for rec in tqdm(records, desc="Grouping by patient"):
        path = rec["modalities"][0]["value"]
        parts = os.path.normpath(path).split(os.sep)

        # MedTrinity case → images/<filename>.jpg (1 image = 1 patient)
        if "images" in parts and parts[-2] == "images":
            patient_id = os.path.basename(path)  # each image gets its own group
        else:
            # Radiopaedia case → use parent folder as patient ID
            patient_id = parts[-3] if len(parts) >= 3 else parts[-2]

        patient_groups[patient_id].append(rec)

    print(f"Total patient groups: {len(patient_groups)}")
    return list(patient_groups.values())

# Greedy split to get 80% of total lines while keeping patients together
def greedy_split(groups, train_ratio):
    random.shuffle(groups)
    total_lines = sum(len(group) for group in groups)
    train_target = int(train_ratio * total_lines)

    train_groups = []
    test_groups = []
    current_train_lines = 0

    for group in groups:
        if current_train_lines + len(group) <= train_target:
            train_groups.append(group)
            current_train_lines += len(group)
        else:
            test_groups.append(group)

    return train_groups, test_groups

if __name__ == "__main__":
    print("Loading combined dataset...")
    data = load_jsonl(combined_input)
    print(f"Total individual records: {len(data)}")

    print("Grouping by patient ID (handling both Radiopaedia and MedTrinity)...")
    grouped_data = group_by_patient(data)

    print("Performing greedy 80/20 split based on number of records...")
    train_groups, test_groups = greedy_split(grouped_data, train_ratio)

    # Flatten groups
    train_records = [item for group in train_groups for item in group]
    test_records = [item for group in test_groups for item in group]

    print(f"Train set: {len(train_records)} records")
    print(f"Test set: {len(test_records)} records")
    final_ratio = len(train_records) / (len(train_records) + len(test_records))
    print(f"Final train ratio: {final_ratio:.2%}")

    # Save output files
    save_jsonl(train_records, train_output)
    save_jsonl(test_records, test_output)
