from openai import OpenAI
import config
import os
from tqdm import tqdm

# 1 - Set client
client = OpenAI(api_key=config.OPEN_API_KEY)

# 2 - List of model folders
model_folders = [
    ("batches_answer_extraction/multimeditron", "output_answer_extraction/multimeditron"),
    ("batches_answer_extraction/gemma", "output_answer_extraction/gemma"),
    ("batches_answer_extraction/qwen", "output_answer_extraction/qwen"),
]

# 3 - for each model 
for BATCHES_FOLDER, OUTPUT_FOLDER in model_folders:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    parts = [x for x in os.listdir(BATCHES_FOLDER) if x.endswith(".jsonl")]

    for part in tqdm(parts, desc=f"Processing {BATCHES_FOLDER}"):
        # 4 - Create a file for the current part
        batch_input_file = client.files.create(
            file=open(f"{BATCHES_FOLDER}/{part}", "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        # 5 - Send a batch
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"LLM as a judge - {part}"},
        )

        # 6 - Save the batch id in a file
        with open(os.path.join(OUTPUT_FOLDER, f"batch_id_{part}.txt"), "w") as f:
            f.write(batch.id)
