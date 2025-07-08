from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List
import config
import json
import os
import re

# List of model paths to process
MODELS_PATHS = [
    "output_answer_extraction/multimeditron",
    "output_answer_extraction/gemma",
    "output_answer_extraction/qwen"
]

FINAL_OUTPUT_FILENAME = "final_extracted_answers.jsonl"

def extract_answer(text: str) -> str:
    match = re.search(r"Answer:\s*([A-E])", text)
    return match.group(1) if match else None

# Load a batch from OpenAI and return the responses as a list of dictionaries
def load_batch(client: OpenAI, batch_id: str) -> List[Dict]:
    batch_status = client.batches.retrieve(batch_id)
    output_file_id = batch_status.output_file_id

    if output_file_id is None:
        raise ValueError(f"Batch {batch_id} has not yet been processed.")
    else:
        binary_response = client.files.content(output_file_id)
        binary_data = binary_response.read()
        text_data = binary_data.decode("utf-8")

        return [json.loads(line) for line in text_data.strip().split("\n") if line.strip()]

# Process each model folder, load the batch IDs, and extract answers
def process_model_folder(API_RESPONSES_PATH: str):
    client = OpenAI(api_key=config.OPEN_API_KEY)
    all_results = []

    for file in os.listdir(API_RESPONSES_PATH):
        if file.startswith("batch_id"):
            part_id = file.split("batch_id_")[1].split(".txt")[0]

            with open(os.path.join(API_RESPONSES_PATH, file), "r") as f:
                batch_openai_id = f.read().strip()

            json_responses = load_batch(client, batch_openai_id)

            for response in json_responses:
                result = {
                    "custom_id": response.get("id"),
                    "answer_extracted": None,
                    "full_response": response
                }

                try:
                    text_output = response["response"]["body"]["choices"][0]["message"]["content"]
                    answer = extract_answer(text_output)
                    result["answer_extracted"] = answer
                except Exception as e:
                    result["error"] = str(e)

                all_results.append(result)

    # Save all results 
    output_file_path = os.path.join(API_RESPONSES_PATH, FINAL_OUTPUT_FILENAME)
    with open(output_file_path, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    print(f"[{API_RESPONSES_PATH}] Saved {len(all_results)} results to {FINAL_OUTPUT_FILENAME}")

def main():
    for API_RESPONSES_PATH in MODELS_PATHS:
        process_model_folder(API_RESPONSES_PATH)

if __name__ == "__main__":
    main()
