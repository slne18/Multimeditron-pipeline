import os
import json
import shutil
from tqdm import tqdm

# JSONL files containing the value for the IRM images
jsonl_path = "/mloscratch/users/noize/IRM_jsonl/MRI_med.jsonl"

# Root folder of medtrinity dataset
source_root = "/mloscratch/users/noize/multimediset_reorganized"

# Destination folder for the copied images
destination_folder = "/mloscratch/users/noize/IRM/images"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Load the JSONL file
with open(jsonl_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Copy the images
copied = 0
already_exists = 0
not_found = 0

# Start progress bar
for item in tqdm(data, desc="IRM images copying"):
    image_value = item["modalities"][0]["value"]
    image_name = os.path.basename(image_value)
    source_path = os.path.join(source_root, image_name)
    destination_path = os.path.join(destination_folder, image_name)

    if os.path.exists(destination_path):
        already_exists += 1  # Image already exists, skip it
    elif os.path.exists(source_path):
        shutil.copyfile(source_path, destination_path)
        copied += 1
    else:
        not_found += 1  # Image not found in source

# Print summary
print(f"\n{copied} images copied, {already_exists} images already existed, {not_found} images not found in source.")