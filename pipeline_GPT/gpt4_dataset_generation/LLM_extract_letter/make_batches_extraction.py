import json
import os
from tqdm import tqdm
import pandas as pd

MAX_TOKENS = 10
MODEL_NAME = "gpt-4o"
TSV_PATH = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"
df_tsv = pd.read_csv(TSV_PATH, sep="\t")

BASE_PROMPT = """You are a multimodal assistant. Based on the explanation and the five answer options provided, extract the most likely letter corresponding to the correct answer (A–E).
Give the final answer strictly in the format: Answer: X where X is a letter.
"""

# List of (JSONL_PATH, OUTPUT_DIR) for each model
model_paths = [
    (
        "/mloscratch/users/mberruye/evaluation/outputs_benchmarks/multimeditron/cleaned_answers.jsonl",
        "batches_answer_extraction/multimeditron"
    ),
    (
        "/mloscratch/users/mberruye/evaluation/outputs_benchmarks/gemma/cleaned_answers.jsonl",
        "batches_answer_extraction/gemma"
    ),
    (
        "/mloscratch/users/mberruye/evaluation/outputs_benchmarks/qwen/cleaned_answers.jsonl",
        "batches_answer_extraction/qwen"
    )
]

def load_data(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [
            json.loads(line)
            for line in f
            if json.loads(line).get("answer") in [None, "", "null", "?"]
        ]

# This function processes each entry and prepares the API request
def process_entry(reasoning: str, options: dict, request_id):
    options_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    prompt = f"{BASE_PROMPT}\n\nChoices:\n{options_str}\n\nReasoning:\n{reasoning}\n\nAnswer:"
    messages = [
        {"role": "system", "content": "You are an expert assistant."},
        {"role": "user", "content": prompt}
    ]
    return {
        "custom_id": f"request-{request_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": MODEL_NAME, "messages": messages, "max_tokens": MAX_TOKENS},
    }

# This function processes the dataset, splitting it into parts based on size
def process_dataset(jsonl_path, output_dir, max_size=0.18 * 1024**3):
    os.makedirs(output_dir, exist_ok=True)
    examples = load_data(jsonl_path)

    file_number = 1
    current_file_size = 0
    current_file_path = os.path.join(output_dir, f"part_{file_number}.jsonl")
    current_file = open(current_file_path, "w")

    for i, entry in tqdm(enumerate(examples), total=len(examples), desc=f"Processing {output_dir}"):
        index = entry["index"]
        row = df_tsv[df_tsv["index"] == index].iloc[0]
        options = {k: row[k] for k in ["A", "B", "C", "D", "E"] if pd.notna(row[k])}

        api_request = process_entry(entry["output"], options, request_id=index)
        json_str = json.dumps(api_request) + "\n"
        json_size = len(json_str.encode("utf-8"))

        if current_file_size + json_size > max_size:
            current_file.close()
            file_number += 1
            current_file_path = os.path.join(output_dir, f"part_{file_number}.jsonl")
            current_file = open(current_file_path, "w")
            current_file_size = 0

        current_file.write(json_str)
        current_file_size += json_size

    current_file.close()
    print(f"Saved {file_number} files to {output_dir}")

if __name__ == "__main__":
    for JSONL_PATH, OUTPUT_DIR in model_paths:
        process_dataset(JSONL_PATH, OUTPUT_DIR)
