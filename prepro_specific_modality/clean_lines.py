from tqdm import tqdm
import json
import os

jsonl_path = "/mloscratch/users/noize/IRM_jsonl/MRI_combined.jsonl"        
new_path = "/mloscratch/users/noize/IRM_jsonl/MRI_combined_clean.jsonl"        

missing_paths = []
total_paths = 0
missing_count = 0

with open(new_path, "w") as f_out:
    with open(jsonl_path, "r") as f_in:
        for line in tqdm(f_in):
            data = json.loads(line)
            modalities = data.get("modalities", [])
            total_paths += len(modalities)

            missing = [x["value"] for x in modalities if not os.path.exists(x["value"])]
            if not missing:
                f_out.write(line)
            else:
                missing_paths.extend(missing)
                missing_count += len(missing)

print(f"Total number of paths: {total_paths}")
print(f"Number of missing paths: {missing_count}")
