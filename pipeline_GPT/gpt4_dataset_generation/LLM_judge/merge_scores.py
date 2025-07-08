import json
import os 

# List of models to process
models = [ "gemma", "qwen"]

# Base directories
answers_base = "/mloscratch/users/mberruye/evaluation/output_benchmarks"
scores_base = "output_eval"

# Process each model
for model in models:
    print(f"Processing model: {model}")

    # Paths to input/output files
    original_path = os.path.join(answers_base, model, "cleaned_answers.jsonl")
    scores_path = os.path.join(scores_base, model, "final_eval.jsonl")
    merged_output_path = os.path.join(scores_base, model, "merged_scored_answers.jsonl")


    # Load evaluation scores: {custom_id → scores}
    id_to_scores = {}
    with open(scores_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "custom_id" in obj and "score_extracted" in obj:
                request_id = obj["custom_id"]
                score = obj["score_extracted"]
                if score:
                    id_to_scores[request_id] = score

    # Merge scores into the original answer entries
    merged = []
    with open(original_path, "r") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            custom_id = f"request-{entry['index']}"
            scores = id_to_scores.get(custom_id)

            if scores:
                entry["score"] = scores

            merged.append(entry)

    # Save the merged output
    with open(merged_output_path, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"Merged scores saved to: {merged_output_path}")
