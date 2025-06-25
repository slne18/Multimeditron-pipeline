import jsonlines
from collections import Counter
import re
import sys
import os
from tqdm import tqdm

unknown_count = 0
unknown_content = []

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else '.'

MAIN_MODALITIES = {"MRI", "CT", "Microscopic", "X-ray", "Ultrasound"}

modality_counter = Counter()
writers = {}

# Compile regex patterns once
modality_patterns = [
    ("MRI", re.compile(r"(?i)T2[- ]FLAIR|T1|T2|FLAIR|MRI")),
    ("CT", re.compile(r"(?i)CT scan|CT|computed tomography")),
    ("Microscopic", re.compile(r"(?i)histological|biopsy|smear|tissue|specimen|hematology|hematologic|close-up|photomicrograph|microscopy|histopathology|histopathological|microscopic|blood smear|high magnification|high-power magnification")),
    ("PET scan", re.compile(r"(?i)PET scan")),
    ("Endoscopic", re.compile(r"(?i)endoscopic view|endoscopy")),
    ("X-ray", re.compile(r"(?i)X-ray|radiograph|radiographic|radiological")),
    ("Mammography", re.compile(r"(?i)mammography|mammogram")),
    ("Ultrasound", re.compile(r"(?i)ultrasound")),
    ("Angiogram", re.compile(r"(?i)angiogram")),
    ("Equipment", re.compile(r"(?i)equipment")),
    ("Diagram", re.compile(r"(?i)diagram")),
    ("General", re.compile(r"(?i)general|follow-up|medical scan|single lesion|photograph|post-treatment control|transverse section|lateral|examination|natural calibratory scan|cranial scan|cross-sectional view")),
]

def extract_modality(description, image_name):
    global unknown_count
    for modality, pattern in modality_patterns:
        if pattern.search(description):
            return modality
    unknown_count += 1
    unknown_content.append({"image_name": image_name, "content": description})
    return "Unknown"

def get_writer(modality):
    if modality not in writers:
        file_path = os.path.join(OUTPUT_PATH, f"{modality}_entries.jsonl")
        writers[modality] = jsonlines.open(file_path, mode='w')
    return writers[modality]


def process_input(path):
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(path, filename)
                process_file(file_path)
    else:
        process_file(path)


def process_file(file_path):
    global modality_counter

    with jsonlines.open(file_path, mode='r') as reader:
        total_lines = sum(1 for _ in reader)

    with jsonlines.open(file_path, mode='r') as reader:
        for i, obj in enumerate(tqdm(reader, total=total_lines, desc=f"Processing {os.path.basename(file_path)}")):
            try:
                if not obj:
                    continue

                modalities = obj.get("modalities", [])
                if not modalities or not isinstance(modalities, list) or not modalities[0].get("value"):
                    continue

                image_name = modalities[0]["value"].strip()

                description = obj.get("text")
                if description is None and "conversations" in obj:
                    for convo in obj["conversations"]:
                        if convo.get("role") == "assistant":
                            description = convo.get("content")
                            break

                if not description:
                    continue

                detected_modality = extract_modality(description, image_name)

                if detected_modality not in MAIN_MODALITIES:
                    detected_modality = "General"

                # Add image path 
                image_path = f"/mloscratch/users/multimediset/{image_name}"
                obj["modalities"][0]["value"] = image_path

                writer = get_writer(detected_modality)
                writer.write(obj)
                modality_counter[detected_modality] += 1

            except Exception as e:
                print(f"Error at line {i+1} in {file_path}: {e}")

# Main Execution
process_input(INPUT_PATH)

# Close all writers
for modality, writer in writers.items():
    writer.close()
    print(f"Finished writing {modality} entries.")

# Save unknown content
if unknown_content:
    unknown_output_file = os.path.join(OUTPUT_PATH, "unknown_content.jsonl")
    with jsonlines.open(unknown_output_file, mode='w') as writer:
        for content in unknown_content:
            writer.write(content)
    print(f"Unknown content saved to {unknown_output_file}")

print(f"Script completed successfully. Total Unknown entries: {unknown_count}")

# Print the number of entries per modality
print("\nNumber of entries per modality:")
for modality, count in modality_counter.items():
    print(f"{modality}: {count}")
