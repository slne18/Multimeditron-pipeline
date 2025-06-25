import json
import os
from tqdm import tqdm
import re


input_path = "/mloscratch/users/noize/medtrinity/separated_modalities/MRI_entries.jsonl"
output_path = "/mloscratch/users/noize/IRM_jsonl/MRI_med.jsonl"

def convert_path(original_path):
    filename = os.path.basename(original_path)
    return f"/mloscratch/users/noize/IRM/images/{filename}"

def clean_text(text):
    text = text.replace("*", "")  #remove *
    text = text.replace("\n", " ")  #replace /n with space
    text = text.replace("\t", " ") #replace /t with space
    text = re.sub(r"\s+", " ", text)  #remove extra spaces
    text = re.sub(r" ?([.,!?;:])", r"\1", text)  #remove spaces before punctuation
    text = text.strip()
    return text

def extract_text_and_modality(json_obj):
    text = ""
    for turn in json_obj.get("conversations", []):
        if turn.get("role") == "assistant":
            text = clean_text(turn.get("content", ""))
            break

    new_modalities = []
    for mod in json_obj.get("modalities", []):
        new_mod = mod.copy()
        if "value" in new_mod:
            new_mod["value"] = convert_path(new_mod["value"])
        new_modalities.append(new_mod)

    return {
        "text": text,
        "modalities": new_modalities
    }


with open(input_path, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

with open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(lines, desc="Conversion in progress"):
        data = json.loads(line)
        simplified = extract_text_and_modality(data)
        outfile.write(json.dumps(simplified, ensure_ascii=False) + "\n")
