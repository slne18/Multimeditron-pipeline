import os
from pathlib import Path

def split_file_in_two(input_file_path):
    # Lire toutes les lignes
    with open(input_file_path, "r") as f:
        lines = f.readlines()

    # Calcul du milieu
    total = len(lines)
    mid = total // 2

    folder = os.path.dirname(input_file_path)
    filename = os.path.basename(input_file_path).replace(".jsonl", "")

    part1 = os.path.join(folder, f"{filename}_part1.jsonl")
    part2 = os.path.join(folder, f"{filename}_part2.jsonl")

    # Sauvegarder les deux parties
    with open(part1, "w") as f1:
        f1.writelines(lines[:mid])

    with open(part2, "w") as f2:
        f2.writelines(lines[mid:])

    print(f"✅ Split done: {input_file_path} → {part1}, {part2}")

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".jsonl") and "_part" not in file:
            full_path = os.path.join(folder_path, file)
            split_file_in_two(full_path)

if __name__ == "__main__":
    process_folder("batches_eval/gemma")
    process_folder("batches_eval/qwen")
