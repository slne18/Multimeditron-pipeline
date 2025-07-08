from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List
import config
import json
import os

# Path to the comparison folder (Gemma vs Qwen)
COMPARISON_PATHS = [
    ("output_comparison/gemma_vs_qwen", "final_comparison.jsonl")
]

def extract_preference_json(text: str) -> Dict[str, str]:
    """
    Extracts preferred model from a response like:
    {
      "preferred_model": "model_a"
    }
    """
    try:
        parsed = json.loads(text)
        if "preferred_model" in parsed:
            return {"preferred_model": parsed["preferred_model"]}
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

def process_comparison_folder(API_RESPONSES_PATH: str, FINAL_OUTPUT_FILE: str, client: OpenAI):
    all_results = []

    for file in os.listdir(API_RESPONSES_PATH):
        if file.startswith("batch_id") and file.endswith(".txt"):
            part_id = file.split("batch_id_")[1].split(".txt")[0]

            with open(os.path.join(API_RESPONSES_PATH, file), "r") as f:
                batch_openai_id = f.read().strip()
            
            try:
                json_responses = load_batch(client, batch_openai_id)
            except ValueError as e:
                print(e)
                continue
            
            for response in json_responses:
                result = {
                    "custom_id": response["id"],
                    "preferred_model": None,
                    "full_response": response
                }

                try:
                    text_output = response["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = extract_preference_json(text_output)
                    result["preferred_model"] = parsed.get("preferred_model")
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
    for API_RESPONSES_PATH, FINAL_OUTPUT_FILE in COMPARISON_PATHS:
        process_comparison_folder(API_RESPONSES_PATH, FINAL_OUTPUT_FILE, client)

if __name__ == "__main__":
    main()
