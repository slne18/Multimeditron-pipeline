import json
import os

def clean_jsonl(file_path):
    # Define the output directory internally
    output_dir = "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries"
    
    # Ensure input file ends with .jsonl
    if not file_path.endswith('.jsonl'):
        raise ValueError("The input file must be a .jsonl file")

    # Get the base name of the file (e.g., data.jsonl -> data_clean.jsonl)
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    clean_file_name = f"{name_without_ext}_clean.jsonl"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create full path to cleaned file
    clean_file_path = os.path.join(output_dir, clean_file_name)

    # Process the file
    with open(file_path, 'r', encoding='utf-8') as infile, open(clean_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                if entry.get("answer") != "?":
                    outfile.write(json.dumps(entry) + '\n')
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")

    print(f"Cleaned file saved to: {clean_file_path}")

clean_jsonl('/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_gemma_no_reflexion.jsonl')
clean_jsonl('/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_gemma.jsonl')
clean_jsonl('/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_Qwen_no_reflexion.jsonl')
clean_jsonl('/mloscratch/users/noize/evaluation/outputs_benchmarks/answers_GMAI-MMBench_2101_Qwen.jsonl')
clean_jsonl('/mloscratch/users/noize/evaluation/outputs_benchmarks/meditron_gpt_Extract.jsonl')