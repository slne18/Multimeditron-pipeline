import sys
sys.path.append("/mloscratch/users/noize/evaluation")

from src.model.model import MultiModalModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from itertools import islice
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import re
from src.model.model import MultimodalRawInput
from src.model.prompt_tokenizers import Llama3PromptTokenizer
from PIL import Image
from collections import Counter

Image.MAX_IMAGE_PIXELS = None
model_path = "/mloscratch/users/noize/evaluation/models/MultiMeditron-Proj-Image"
model_answers_path = "/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_meditron_no_reflexion.txt"

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load the benchmark dataset
dataset_tsv = "/mloscratch/users/noize/evaluation/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"
dataset_images = "/mloscratch/users/noize/evaluation/GMAI-MMBench/images/GMAI-MMBench_VAL/"

# Load dataset
dataset = Dataset.from_pandas(pd.read_csv(dataset_tsv, sep="\t", header=0))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

# Load model
model = MultiModalModelForCausalLM.from_pretrained(model_path)
model.eval()
model.to("cuda")
modalities_num_embeddings = {x.modality_type: x.num_patches_per_entry for x in model.config.modalities}

# Prompt tokenizer
prompt_tokenizer = Llama3PromptTokenizer(
    tokenizer=tokenizer,
    modalities_num_embeddings=modalities_num_embeddings,
    attachment_token_idx=attachment_token_idx
)

if not os.path.exists("outputs_benchmarks"):
    os.mkdir("outputs_benchmarks")

if os.path.exists(model_answers_path):
    with open(model_answers_path, "r") as f:
        answers = list(f.read())
else:
    answers = []

# Process a batch
def process_batch(batch, global_index_start):
    modalities_batch = [
        [{"type": "image", "value": os.path.join(dataset_images, f"{index}.jpg")}]
        for index in batch["index"]
    ]

    conversations_batch = [
        [
            {
                "role": "user",
                "content": (
                    f"{ATTACHMENT_TOKEN} {question}\n\n"
                    + "\n".join(f"{letter}. {opt}" for letter in "ABCDE" if (opt := options.get(letter)) is not None)
                    + "\nPlease answer by only clearly stating the final answer in the form 'Answer: X' where X is one of A, B, C, D, or E."
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
            temperature=0.5, do_sample=True, max_length=150
        )

    batch_answers = []
    for idx, output in enumerate(outputs_batch):
        rep = tokenizer.decode(output, skip_special_tokens=True).replace("**", "")
        print(f"\n--- Question {global_index_start + idx} ---")
        print(rep)

        extracted_answer = "?"
        match = re.search(r"(?i)answer\s*[:\-–]\s*([A-E])\b", rep)
        if match:
            extracted_answer = match.group(1)
        else:
            match = re.search(r"(?i)(correct|right)\s+(option|answer)\s+(is|:)?\s*([A-E])[\.\s]", rep)
            if match:
                extracted_answer = match.group(4)
        if not match:
            match = re.search(r"(?i)is\s+([A-E])[\.\n\s]", rep)
            if match:
                extracted_answer = match.group(1)
        if not match:
            match = re.search(r"^[A-E]$", rep.strip(), re.MULTILINE)
            if match:
                extracted_answer = match.group(0)
        if not match:
            letters = re.findall(r"\b[A-E]\b", rep)
            if letters:
                most_common = Counter(letters).most_common(1)[0][0]
                extracted_answer = most_common

        batch_answers.append(extracted_answer)

    return batch_answers

# Run through batches
batch_size = 16
for i in tqdm(range(len(answers), len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    batch_answers = process_batch(batch, i)
    answers.extend(batch_answers)

    with open(model_answers_path, "w") as f:
        f.write("".join(answers))
