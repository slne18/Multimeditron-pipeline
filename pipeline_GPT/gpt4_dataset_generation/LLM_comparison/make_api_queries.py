from openai import OpenAI
import config
import os
from tqdm import tqdm

# 1 - Set client
client = OpenAI(api_key=config.OPEN_API_KEY)

# Folder for pairwise comparison between Gemma and Qwen
BATCHES_FOLDER = "batches_comparison/gemma_vs_qwen"
OUTPUT_FOLDER = "output_comparison/gemma_vs_qwen"

# 3 - for each model 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
parts = [x for x in os.listdir(BATCHES_FOLDER) if x.endswith(".jsonl")]

for part in tqdm(parts):
    # 2 - Create a file for the current part
    batch_input_file = client.files.create(
        file=open(f"{BATCHES_FOLDER}/{part}", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    # Create the batch job
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Comparison Gemma vs Qwen - {part}"},
    )

    # Save the batch id in a file
    with open(os.path.join(OUTPUT_FOLDER, f"batch_id_{part}.txt"), "w") as f:
        f.write(batch.id)
