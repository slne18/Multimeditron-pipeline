import os
import json
import re
from tqdm import tqdm

# Paths
dataset_root = "/mloscratch/users/noize/IRM/image/Ct"
output_file = "/mloscratch/users/noize/IRM_jsonl/IRM_radiopedia.jsonl"

def clean_text(text):
    text = text.replace("*", "")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" ?([.,!?;:])", r"\1", text)
    return text.strip()


def get_all_patient_data(root_dir):
    data = []

    # Liste des dossiers patients
    patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for patient_folder in tqdm(patient_folders, desc="Treatment in progress"):
        patient_folder_path = os.path.join(root_dir, patient_folder)

        for subfolder in os.listdir(patient_folder_path):
            subfolder_path = os.path.join(patient_folder_path, subfolder)
            image_path = os.path.join(subfolder_path, "images.png")
            report_path = os.path.join(subfolder_path, "report.txt")

            if os.path.exists(image_path) and os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read().strip()
                    report_content = clean_text(raw_content)

                record = {
                    "text": report_content,
                    "modalities": [{
                        "type": "image",
                        "value": image_path  # tu peux remplacer par f"images/{os.path.basename(image_path)}" si tu préfères
                    }]
                }
                data.append(record)

    return data

def save_as_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = get_all_patient_data(dataset_root)

    if not dataset:
        raise FileNotFoundError(f"No data found {dataset_root}")

    save_as_jsonl(dataset, output_file)

    print(f"Data saved : {output_file}")
