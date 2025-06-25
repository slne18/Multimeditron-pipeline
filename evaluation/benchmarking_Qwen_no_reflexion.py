import sys
sys.path.append("/mloscratch/users/noize/evaluation")

from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import os
import re
from PIL import Image
from collections import Counter
import json
import torch

Image.MAX_IMAGE_PIXELS = None

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_answers_path = "/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_Qwen_no_reflexion.jsonl"
dataset_tsv = "/mloscratch/users/noize/evaluation/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"
dataset_images = "/mloscratch/users/noize/evaluation/GMAI-MMBench/images/GMAI-MMBench_VAL/"

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load dataset
dataset = Dataset.from_pandas(pd.read_csv(dataset_tsv, sep="\t", header=0))

# Load Qwen pipeline
pipe = pipeline(
    task="image-text-to-text",
    model=model_path,
    device=0,
    torch_dtype=torch.bfloat16
)

if not os.path.exists("outputs_benchmarks"):
    os.mkdir("outputs_benchmarks")

answers = []


def make_conversations_batch(batch, modalities_batch):
    for modalities in modalities_batch:
        for mod in modalities:
            mod["path"] = mod.pop("value")

    return [
        [
            {
                "role": "user",
                "content": modalities + [
                    {"type": "text", "text": (
                        f"{question}\n\n"
                        + "\n".join(f"{letter}. {opt}" for letter in "ABCDE" if (opt := options.get(letter)) is not None)
                        + "\nPlease answer by only clearly stating the final answer in the form 'Answer: X' where X is one of A, B, C, D, or E.")
                    }
                ]
            },
        ]
        for question, options, modalities in zip(
            batch["question"],
            [{letter: batch[letter][i] for letter in "ABCDE"} for i in range(len(batch["question"]))],
            modalities_batch
        )
    ]

def extract_answer(rep):
    from collections import Counter
    import re

    rep = rep.replace("**", "").strip()

    # Look for explicit "Answer: A" pattern
    match = re.search(r"(?i)\banswer\s*[:\-–]?\s*([A-E])\b", rep)
    if match:
        return match.group(1), rep

    # Look for "Correct option is A" or similar
    match = re.search(r"(?i)\b(correct|right)\s+(option|answer)\s+(is|:)?\s*([A-E])\b", rep)
    if match:
        return match.group(4), rep

    return "?", rep

def process_batch(batch, global_index_start):
    modalities_batch = [
        [{"type": "image", "value": os.path.join(dataset_images, f"{index}.jpg")}] for index in batch["index"]
    ]
    conversations_batch = make_conversations_batch(batch, modalities_batch)
    outputs_batch = [x[0]["generated_text"] for x in pipe(text=conversations_batch, max_new_tokens=200, return_full_text=False)]

    batch_entries = []
    for idx, output in enumerate(outputs_batch):
        answer, reasoning = extract_answer(output)
        entry = {
            "index": batch["index"][idx],
            "question": batch["question"][idx],
            "output": reasoning.strip(),
            "answer": answer
        }
        batch_entries.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    return batch_entries

# Run through batches
batch_size = 16
for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    batch_entries = process_batch(batch, i)
    answers.extend(batch_entries)

    with open(model_answers_path, "w") as f:
        for entry in answers:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")