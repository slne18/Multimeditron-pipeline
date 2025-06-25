import argparse
import jsonlines
import os
import glob
from itertools import combinations
from difflib import SequenceMatcher
from collections import defaultdict

# Argument parser to take file paths as input
parser = argparse.ArgumentParser(description="Analyze JSONL files for similarity.")
parser.add_argument('files', nargs='+', help='List of JSONL files or directories to process') 
args = parser.parse_args()

# Handle folder inputs: expand to include all .jsonl files
all_files = []
for path in args.files:
    if os.path.isdir(path):  
        all_files.extend(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        all_files.append(path)

args.files = all_files  # Update file list

# Store extracted descriptions
descriptions = []

# Function to extract the description from JSONL objects
def extract_description(obj):
    if "conversations" in obj and isinstance(obj["conversations"], list):
        for convo in obj["conversations"]:
            if convo.get("role") == "assistant":
                return convo.get("content", "").strip()
    return None

# Read descriptions from JSONL files
for file_path in args.files:
    if file_path.endswith(".jsonl"):
        try:
            with jsonlines.open(file_path, mode='r') as reader:
                for obj in reader:
                    description = extract_description(obj)
                    if description:
                        descriptions.append(description)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

# Compute similarity and find duplicates
if len(descriptions) > 1:
    similarity_scores = []
    exact_match_groups = defaultdict(list)

    for a, b in combinations(descriptions, 2):
        score = SequenceMatcher(None, a, b).ratio()
        similarity_scores.append(score)

        if score == 1.0:  # 100% similarity
            exact_match_groups[a].append(b)

    # Print groups of captions that are 100% similar
    if exact_match_groups:
        print("\n🔍 Groups of Identical Captions:")
        for key, duplicates in exact_match_groups.items():
            print("\n--- Identical Captions ---")
            print(key)
            for caption in duplicates:
                print(caption)

    # Print average similarity score
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    print(f"\n📊 Average Similarity Score: {avg_similarity:.2%}")

else:
    print("Not enough descriptions to compute similarity.")
