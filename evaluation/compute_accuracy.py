import json
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict


# Paths
prediction_paths = {
    "Gemma reasoning": "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries/answers_GMAI-MMBench_2101_gemma_clean.jsonl",
    "Gemma no reasoning": "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries/answers_GMAI-MMBench_2101_gemma_no_reflexion_clean.jsonl",
    "Qwen reasoning": "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries/answers_GMAI-MMBench_2101_Qwen_clean.jsonl",
    "Qwen no reasoning": "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries/answers_GMAI-MMBench_2101_Qwen_no_reflexion_clean.jsonl",
    "Meditron reasoning": "/mloscratch/users/noize/evaluation/outputs_benchmarks/clean_missing_entries/meditron_gpt_Extract_clean.jsonl",
}

dataset_tsv = "/mloscratch/users/noize/evaluation/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"

# Load data
dataset = pd.read_csv(dataset_tsv, sep="\t")
correct_answers = {row["index"]: row["answer"] for _, row in dataset.iterrows()}

# Metrics folder
metrics_dir = "metrics_eval"
os.makedirs(metrics_dir, exist_ok=True)

# Compute accuracy & precision
metrics = {}
for model_name, path in prediction_paths.items():
    with open(path, "r") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    total, correct = 0, 0
    pred_counts, correct_pred_counts = defaultdict(int), defaultdict(int)

    for pred in predictions:
        idx = pred["index"]
        pred_answer = pred["answer"]
        gold_answer = correct_answers.get(idx)
        if gold_answer:
            total += 1
            pred_counts[pred_answer] += 1
            if pred_answer == gold_answer:
                correct += 1
                correct_pred_counts[pred_answer] += 1

    acc = correct / total if total > 0 else 0
    precision_per_class = [correct_pred_counts[cls] / pred_counts[cls] for cls in pred_counts]
    prec = sum(precision_per_class) / len(precision_per_class) if precision_per_class else 0

    metrics[model_name] = {"accuracy": acc, "precision": prec, "correct": correct, "total": total}

# Add manual entries
manual_metrics = {
    "GPT-4o reasoning": {"accuracy": 0.341, "precision": 0.521},
    "GPT-4o no reasoning": {"accuracy": 0.555, "precision": 0.559},
    "MedGEMa reasoning": {"accuracy": 0.434, "precision": 0.467},
    "MedGEMa no reasoning": {"accuracy": 0.455, "precision": 0.455},
}
for name, vals in manual_metrics.items():
    metrics[name] = {**vals, "correct": 0, "total": 0}

# Plot setup
comparison_groups = {
    "Gemma": ["Gemma reasoning", "Gemma no reasoning"],
    "Qwen": ["Qwen reasoning", "Qwen no reasoning"],
    "GPT-4o": ["GPT-4o reasoning", "GPT-4o no reasoning"],
    "MedGEMa": ["MedGEMa reasoning", "MedGEMa no reasoning"],
}
colors = {
    "accuracy reasoning": "#1f77b4",    # dark blue
    "accuracy no reasoning": "#aec7e8", # light blue
    "precision reasoning": "#2ca02c",   # dark green
    "precision no reasoning": "#98df8a" # light green
}

# Plot per model
for model_base, variants in comparison_groups.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_labels = []
    values = []
    bar_colors = []

    for v in variants:
        acc = metrics[v]["accuracy"]
        prec = metrics[v]["precision"]
        reasoning = "no reasoning" not in v  # <-- correctly distinguishes

        values += [acc, prec]
        bar_labels += [
            "Reasoning Accuracy" if reasoning else "No Reasoning Accuracy",
            "Reasoning Precision" if reasoning else "No Reasoning Precision"
        ]
        bar_colors += [
            colors["accuracy reasoning" if reasoning else "accuracy no reasoning"],
            colors["precision reasoning" if reasoning else "precision no reasoning"]
        ]

    bars = ax.bar(bar_labels, values, color=bar_colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.2%}", ha='center', va='bottom', fontsize=8)

    ax.set_title(f"{model_base}: Accuracy & Precision")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{model_base.lower()}_accuracy_precision.png"))
    plt.close()


# Plot Meditron separately
if "Meditron reasoning" in metrics:
    m = metrics["Meditron reasoning"]
    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(["Accuracy", "Precision"], [m["accuracy"], m["precision"]],
                  color=[colors["accuracy reasoning"], colors["precision reasoning"]])
    for bar, val in zip(bars, [m["accuracy"], m["precision"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2%}", ha='center')
    ax.set_title("Meditron (Reasoning)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "meditron_accuracy_precision.png"))
    plt.close()