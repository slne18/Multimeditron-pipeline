import os
import random
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

train_rate = 0.8 

folders = ((count_subfolders(path)))
train_folders = random.sample(folders, math.floor(train_rate*len(folders)))
test_folders = [x for x in folders if x not in train_folders]

dataset_train = get_list(train_folders)
dataset_test = get_list(test_folders)


with open(path + "/Ultrasound_train.jsonl", "w", encoding="utf-8") as f:
    for record in dataset_train:
        f.write(record + "\n")

dataset_test = get_list(test_folders)
with open(path + "/Ultrasound_test.jsonl", "w", encoding="utf-8") as f:
    for record in dataset_test:
        f.write((record) + "\n")

