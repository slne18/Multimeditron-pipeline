import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
MODELS = ["gemma", "qwen", "multimeditron"]
GROUP_COLUMNS = ["category", "clinical VQA task", "department", "modality"]
TSV_PATH = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"
ANSWERS_BASEDIR = "output_answer_extraction"
OUTPUT_DIR = "visualizations/accuracy_by_group"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ground truth
df_gt = pd.read_csv(TSV_PATH, sep="\t")
df_gt["index"] = df_gt["index"].astype(str)
df_gt = df_gt[["index", "answer"] + GROUP_COLUMNS].rename(columns={"answer": "correct_answer"})

# Merge accuracy per model
model_dfs = {}
for model in MODELS:
    path = os.path.join(ANSWERS_BASEDIR, model, "merged_answers.jsonl")
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    df_model = pd.DataFrame(data)
    df_model["index"] = df_model["index"].astype(str)
    df_model = df_model[["index", "answer"]].rename(columns={"answer": "predicted_answer"})
    merged = df_gt.merge(df_model, on="index", how="inner")
    merged["is_correct"] = merged["predicted_answer"] == merged["correct_answer"]
    model_dfs[model] = merged

# Compute and plot accuracy by group
for group in GROUP_COLUMNS:
    group_accs = []
    for model, df in model_dfs.items():
        acc_by_group = df.groupby(group)["is_correct"].mean().reset_index()
        acc_by_group["model"] = model
        group_accs.append(acc_by_group)

    all_group_acc = pd.concat(group_accs)

    plt.figure(figsize=(12, max(4, len(all_group_acc[group].unique()) * 0.4)))
    sns.barplot(data=all_group_acc, x=group, y="is_correct", hue="model")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Accuracy by {group}")
    plt.ylabel("Accuracy")
    plt.xlabel(group)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"accuracy_by_{group.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[✓] Saved plot: {plot_path}")
