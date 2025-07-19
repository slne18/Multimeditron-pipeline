import os
import math
import json
import pyarrow as pa

path = "/mloscratch/users/deschryv/clipFineTune/Ultrasound"

def count_subfolders(path):
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]

def get_list(folder_list):
    dataset = []
    for folder in folder_list:
        for sub_folder in count_subfolders(path + "/" + folder):
            with open(path + "/"+ folder +"/" + sub_folder +"/report.txt") as f:
                text = f.read().replace("\n", "").replace('"', "").replace("\\", "").replace("*", "")
                dico = {"text": text, "modalities": [{"type": "image", "value": path + "/" + folder+"/" + sub_folder + "/images.png"}]}
                dataset.append(json.dumps(dico))
                    
    
    return dataset

folders = ((count_subfolders(path)))
print(len(folders))
dataset = get_list(folders)
with open(path + "/Ultrasound_all.jsonl", "w", encoding="utf-8") as f:
    for record in dataset:
        f.write(record + "\n")
