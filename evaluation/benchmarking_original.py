import sys
sys.path.append("../") #to import modules from the rest of the repo

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
Image.MAX_IMAGE_PIXELS = None 

model_path = "/mloscratch/users/mberruye/evaluation/models/MultiMeditron-Proj-Image"

#model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_answers_path = "/mloscratch/users/mberruye/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101.txt"

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load the benchmark dataset
dataset_tsv = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"
dataset_images = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/images/GMAI-MMBench_VAL/"

dataset = Dataset.from_pandas(pd.read_csv(dataset_tsv, sep="\t", header=0))

batch_size = 16
batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

model = MultiModalModelForCausalLM.from_pretrained(model_path)
model.to("cuda")
#print("Modalities:", [m.modality_type for m in model.config.modalities])
modalities_num_embeddings = {x.modality_type: x.num_patches_per_entry for x in model.config.modalities}

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

# Function to process a batch
def process_batch(batch):
    # Prepare modalities for the batch
    modalities_batch = [
        [{"type": "image", "value": os.path.join(dataset_images, f"{index}.jpg")}]
        for index in batch["index"]
    ]

    # Prepare conversations for the batch
    conversations_batch = [
        [
            {
                "role": "user",
                "content": (
                    # "You are an expert medical assistant who is passing a multiple-choice exam on medical questions. "
                    # "You will be prompted with an attachment, a question and several options marked with letters A, B, C, D or E. "
                    "<|reserved_special_token_0|> " + question + "\n\n"
                    "\n".join(f"{letter}. {opt}" for letter in "ABCDE" if (opt := options.get(letter)) is not None)
                    + "Let's think step by step."
                ),
            },
        ]
        for question, options in zip(batch["question"], [
            {letter: batch[letter][i] for letter in "ABCDE"} for i in range(len(batch["question"]))
        ])
    ]

    # Tokenize inputs for the batch
    inputs_batch = [
        prompt_tokenizer.tokenize_conversation(conversations, modalities, add_eos_token=False, add_generation_prompt=True)
        for conversations, modalities in zip(conversations_batch, modalities_batch)
    ]

    # Extract input IDs and pad them
    input_ids_list = [torch.tensor(inputs["input_ids"]) for inputs in inputs_batch]
    input_ids_batch = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
    # Prepare multimodal inputs
    multimodal_inputs_batch = [modalities for modalities in modalities_batch]

    # Generate outputs
    with torch.no_grad():
        outputs_batch = model.generate(
            input_ids_batch, multimodal_inputs=multimodal_inputs_batch,
            temperature=0.5, do_sample=True
        )

    # Decode and extract answers
    batch_answers = []
    for output in outputs_batch:
        rep = tokenizer.decode(output).replace("**", "")
        print(rep)
        try:
            extracted_answer = rep[rep.rindex("Answer: ") + len("Answer: ")][0]
        except:
            extracted_answer = "?"
        batch_answers.append(extracted_answer)

    return batch_answers

# Iterate through the dataset in batches
for i in tqdm(range(len(answers), len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]  # Slice the dataset
    batch_answers = process_batch(batch)
    answers.extend(batch_answers)

    # Save progress after each batch
    with open(model_answers_path, "w") as f:
        f.write("".join(answers))
