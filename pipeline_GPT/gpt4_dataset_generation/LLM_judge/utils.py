'''
File name: utils.py
Author: Maud Dupont-Roc & Fabrice Nemo
Date created: 03/08/2024
Date last modified: 07/05/2025
Python Version: 3.10
'''

# Import libraries
import json
from openai import OpenAI
import os
import base64

from typing import List, Tuple, Optional

# path of the jsonl of the dataset
base_jsonl = "/mloscratch/users/nemo/datasets/US_data/US_SEG/US_SEG.jsonl"
# base of the paths of the images (so that the path in the jsonl can be concatenated to it)
base_folder_img = "/mloscratch/users/nemo/datasets/US_data/US_SEG/"

from tqdm import tqdm

#load data from the jsonl
#nb_samples is the number of samples to load. it can be left as None to mean "load the whole dataset"
def load_data_raw(nb_samples: Optional[int] = None) -> List[str]:
    with open(base_jsonl, "r") as f:
        if nb_samples is None:
            return list(json.loads(line) for line in f)
        else:
            return list(json.loads(line) for i, line in zip(range(nb_samples), f))

def encode_img(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#load the dataset from standard format and format it into [(text1, base64image1), (text2, base64image2)]
#assuming all the modalities are images
#nb_samples is the number of samples to load. it can be left as None to mean "load the whole dataset"
def load_data(nb_samples: Optional[int] = None):
    lines_new = []
    for line in tqdm(load_data_raw(nb_samples)):
        text = line["text"]
        image_paths = [os.path.join(base_folder_img, mod["value"]) for mod in line["modalities"]]

        # Encode images and append them as a tuple
        encoded_images = tuple(encode_img(path) for path in image_paths)
        lines_new.append((text, encoded_images))

    return lines_new