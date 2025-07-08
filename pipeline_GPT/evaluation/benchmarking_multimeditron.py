import sys
sys.path.append("../")  # to import modules from the rest of the repo

from src.model.model import MultiModalModelForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset
from itertools import islice
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from src.model.model import MultimodalRawInput
from src.model.prompt_tokenizers import Llama3PromptTokenizer
from PIL import Image
import re
import json
import base64
from io import BytesIO

Image.MAX_IMAGE_PIXELS = None

model_path = "/mloscratch/users/mberruye/evaluation/models/MultiMeditron-Proj-Image"
model_answers_path = "/mloscratch/users/mberruye/evaluation/outputs_benchmarks/multimeditron/answers_GMAI-MMBench_2101_reflexion.txt"

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load the benchmark dataset
dataset_tsv = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"

dataset = Dataset.from_pandas(pd.read_csv(dataset_tsv, sep="\t", header=0))

batch_size = 16
batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

# Load model
model = MultiModalModelForCausalLM.from_pretrained(model_path)
model.to("cuda")

modalities_num_embeddings = {x.modality_type: x.num_patches_per_entry for x in model.config.modalities}

prompt_tokenizer = Llama3PromptTokenizer(
    tokenizer=tokenizer,
    modalities_num_embeddings=modalities_num_embeddings,
    attachment_token_idx=attachment_token_idx
)

if not os.path.exists("outputs_benchmarks"):
    os.mkdir("outputs_benchmarks")


answers = []

# Clean output and extract final answer
def clean_output_and_extract(output):
    output = re.sub(r'(<\|eot_id\|>)+', '', output)
    output = re.sub(r'<\|reserved_special_token_\d+\|>', '', output)
    output = output.strip()
    match = re.findall(r'Answer\s*[:：]\s*([ABCDE])', output)
    answer = match[-1] if match else None
    reasoning = output.split(f"Answer: {answer}")[0].strip() if answer else output
    return reasoning, answer

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

# Process batch
def process_batch(batch):
    modalities_batch = []
    for base64_str in batch["image"]:
        img = decode_base64_image(base64_str)
        modalities_batch.append([{"type": "image", "value": img}])


    conversations_batch = [
        [
            {
                "role": "user",
                "content": (
                    ATTACHMENT_TOKEN + " " + question + "\n\n"
                    + "\n".join(f"{letter}. {opt}" for letter in "ABCDE" if (opt := options.get(letter)) is not None)
                    + "\nPlease explain your reasoning step by step and conclude by clearly stating the final answer in the form 'Answer: X' where X is one of A, B, C, D, or E."
                ),
            },
        ]
        for question, options in zip(batch["question"], [
            {letter: batch[letter][i] for letter in "ABCDE"} for i in range(len(batch["question"]))
        ])
    ]

    inputs_batch = [
        prompt_tokenizer.tokenize_conversation(conversations, modalities, add_eos_token=False, add_generation_prompt=True)
        for conversations, modalities in zip(conversations_batch, modalities_batch)
    ]

    input_ids_list = [torch.tensor(inputs["input_ids"]) for inputs in inputs_batch]
    input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
    multimodal_inputs_batch = [modalities for modalities in modalities_batch]

    with torch.no_grad():
        outputs_batch = model.generate(
            input_ids_batch, multimodal_inputs=multimodal_inputs_batch,
            temperature=0.5, do_sample=True
        )

    batch_outputs = [tokenizer.decode(output).replace("**", "") for output in outputs_batch]
    batch_json_entries = []

    for i, rep in enumerate(batch_outputs):
        reasoning, final_answer = clean_output_and_extract(rep)
        entry = {
            "index": batch["index"][i],
            "question": batch["question"][i],
            "output": reasoning,
            "answer": final_answer,
        }
        batch_json_entries.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    return batch_json_entries

# Iterate through batches
for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    batch_entries = process_batch(batch)
    answers.extend(batch_entries)

    os.makedirs(os.path.dirname(model_answers_path), exist_ok=True)
    with open(model_answers_path, "w") as f:
        for entry in answers:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
