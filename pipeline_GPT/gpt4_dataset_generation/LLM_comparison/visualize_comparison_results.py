import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_PATH = "output_comparison/gemma_vs_qwen/final_comparison.jsonl"
TSV_PATH = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"
GROUP_COLUMNS = ["category", "clinical VQA task", "department", "modality"]
TITLE_MAPPING = {
    "category": "Organ",
    "clinical VQA": "Clinical Task",
    "departement": "Department",
    "Modality": "Imaging Modality"
}
OUTPUT_DIR = "visualizations/heatmaps_by_group"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model comparison results
with open(RESULTS_PATH, "r") as f:
    responses = [json.loads(line) for line in f if line.strip()]

# Map index → preferred_model
index_to_pref = {
    r["full_response"]["custom_id"].split("-")[1]: r["preferred_model"]
    for r in responses
    if "full_response" in r and r.get("preferred_model") in {"model_a", "model_b", "tie"}
}

# Load TSV and merge preference data
df = pd.read_csv(TSV_PATH, sep="\t")
df["index"] = df["index"].astype(str)
df["preferred_model"] = df["index"].map(index_to_pref)
df = df.dropna(subset=["preferred_model"])


for col in GROUP_COLUMNS:
    if col not in df.columns:
        print(f"Column '{col}' not found in TSV.")
        continue

    pivot = df.groupby([col, "preferred_model"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(10, max(4, 0.4 * len(pivot_pct))))
    sns.heatmap(pivot_pct, annot=True, fmt=".2f", cmap="Blues", cbar=True)

    title = TITLE_MAPPING.get(col, col)
    plt.title(f"Model preferences by {title}")
    plt.xlabel("Preferred model (GPT-4o judgment)")
    plt.ylabel(title)
    plt.tight_layout()

    # Save heatmap
    filename = f"heatmap_by_{col.replace(' ', '_').lower()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Saved heatmap to: {filepath}")
