from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List
import config
import json
import os
import re

# Paths to the model output directories and final output files
MODEL_PATHS = [
    #("output_eval/multimeditron", "final_eval.jsonl"),
    ("output_eval/gemma", "final_eval.jsonl"),
    ("output_eval/qwen", "final_eval.jsonl")
]

EVAL_CRITERIA = [
    "accuracy",
    "completeness",
    "instruction_following",
    "communication",
    "context_awareness"
]

def extract_scores_json(text: str) -> Dict[str, int]:
    """
    Extracts scores for clarity, relevance, accuracy, justification, and structure from a JSON string.
    The expected format is strictly:
    {
      "accuracy": X,
      "completeness": X,
      "instruction_following": X,
      "communication": X,
      "context_awareness": X
    }
    """
    try:
        parsed = json.loads(text)
        if all(k in parsed for k in EVAL_CRITERIA):
            return {k: int(parsed[k]) for k in EVAL_CRITERIA}
    except Exception:
        pass
    return {}

    

def load_batch(client: OpenAI, batch_id: str) -> List[Dict]:
    batch_status = client.batches.retrieve(batch_id)
    output_file_id = batch_status.output_file_id

    if output_file_id is None:
        raise ValueError(f"Batch {batch_id} has not yet been processed, wait and try again later")
    else:
        binary_response = client.files.content(output_file_id)
        binary_data = binary_response.read()
        text_data = binary_data.decode("utf-8")

        return [json.loads(line) for line in text_data.strip().split("\n") if line.strip()]


def process_model_folder(API_RESPONSES_PATH: str, FINAL_OUTPUT_FILE: str, client: OpenAI):
    all_results = []

    for file in os.listdir(API_RESPONSES_PATH):
        if file.startswith("batch_id"):
            part_id = file.split("batch_id_")[1].split(".jsonl")[0]

            with open(os.path.join(API_RESPONSES_PATH, file), "r") as f:
                batch_openai_id = f.read()
            
            try:
                json_responses = load_batch(client, batch_openai_id)
            except ValueError as e:
                print(e)
                continue
            
            for response in json_responses:
                result = {
                    "custom_id": response["id"],
                    "scores": None,
                    "full_response": response
                }

                try:
                    text_output = response["response"]["body"]["choices"][0]["message"]["content"]
                    scores = extract_scores_json(text_output)
                    result["scores"] = scores
                except Exception as e:
                    result["error"] = str(e)

                all_results.append(result)

    output_path = os.path.join(API_RESPONSES_PATH, FINAL_OUTPUT_FILE)
    with open(output_path, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    print(f"[{API_RESPONSES_PATH}] Saved {len(all_results)} results to {FINAL_OUTPUT_FILE}")


def main():
    client = OpenAI(api_key=config.OPEN_API_KEY)
    for API_RESPONSES_PATH, FINAL_OUTPUT_FILE in MODEL_PATHS:
        process_model_folder(API_RESPONSES_PATH, FINAL_OUTPUT_FILE, client)

if __name__ == "__main__":
    main()