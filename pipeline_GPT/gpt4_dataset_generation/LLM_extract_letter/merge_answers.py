import json
import os

# List of models to process
models = ["multimeditron", "gemma", "qwen"]

# Root directories for input/output files
base_cleaned = "/mloscratch/users/mberruye/evaluation/outputs_benchmarks"
base_extracted = "output_answer_extraction"


for model in models:
    print(f"\nProcessing model: {model}")

    cleaned_answers_path = os.path.join(base_cleaned, model, "cleaned_answers.jsonl")
    extracted_answers_path = os.path.join(base_extracted, model, "final_extracted_answers.jsonl")
    merged_output_path = os.path.join(base_extracted, model, "merged_answers.jsonl")

    # Load extracted answers
    id_to_answer = {}
    missing_extracted_count = 0

    with open(extracted_answers_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry.get("full_response", {}).get("custom_id")
            answer = entry.get("answer_extracted")
            # verify that answer are not None
            if custom_id and answer:
                id_to_answer[custom_id] = answer
            # Count missing answers
            elif custom_id:
                missing_extracted_count += 1 


    # Merge with cleaned_answers.jsonl
    merged_entries = []
    with open(cleaned_answers_path, "r") as f:
        for i, line in enumerate(f, start=1):
            entry = json.loads(line)

            if entry.get("answer") is None:
                custom_id = f"request-{entry['index']}"
                new_answer = id_to_answer.get(custom_id)
                if new_answer:
                    entry["answer"] = new_answer

            merged_entries.append(entry)

    # Save the merged file
    with open(merged_output_path, "w") as f:
        for entry in merged_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Merged answers saved to: {merged_output_path}")
    print(f"No letter extracted for {missing_extracted_count} responses.")
