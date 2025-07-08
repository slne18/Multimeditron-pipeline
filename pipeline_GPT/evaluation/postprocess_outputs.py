import json
import re
import os
from pathlib import Path

def extract_answer(text):
    match = re.search(r"[Aa]nswer\s*[:：]?\s*([A-E])", text)
    if match:
        return match.group(1)
    return None

def clean_jsonl_file(input_path):
    cleaned = []

    with open(input_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Clean the 'output' field
            output_text = entry.get("output", "").strip()
            if not output_text or output_text in ["<|start_header_id|>", "<|start_header_id|>\n"]:
                continue

            # Remove unwanted tags and clean up the text
            for tag in ["assistant<|end_header_id|>", "<|start_header_id|>", "<|end_header_id|>"]:
                output_text = output_text.replace(tag, "")

            # Remove special tokens and clean up the text
            output_text = output_text.replace("**", "").replace("\n", "").strip()
            entry["output"] = output_text

            if entry.get("answer") in [None, "", "null", "?"]:
                entry["answer"] = extract_answer(output_text)

            cleaned.append(entry)

    # Save the cleaned entries to a new JSONL file
    output_path = input_path.with_name("cleaned_answers.jsonl")
    with open(output_path, "w") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

# Main script to clean JSONL files in specified directories
model_dirs = [
    Path("outputs_benchmarks/multimeditron"),
    Path("outputs_benchmarks/qwen"),
    Path("outputs_benchmarks/gemma"),
]

for model_dir in model_dirs:
    for ext in ("*.jsonl", "*.txt"):
        for file in model_dir.glob(ext):
            clean_jsonl_file(file)