import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths to merged prediction files for each model
prediction_paths = {
    "MultiMeditron": "output_answer_extraction/multimeditron/merged_answers.jsonl",
    "Gemma": "output_answer_extraction/gemma/merged_answers.jsonl",
    "Qwen": "output_answer_extraction/qwen/merged_answers.jsonl",
}

# Path to the reference TSV file containing ground-truth answers
dataset_tsv = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"
df = pd.read_csv(dataset_tsv, sep="\t")
correct_answers = dict(zip(df["index"], df["answer"]))

# Create directory to store evaluation metrics and mismatches
metrics_dir = "metrics_eval"
os.makedirs(metrics_dir, exist_ok=True)

# Evaluate accuracy for each model
accuracies = {}
for model_name, path in prediction_paths.items():
    with open(path, "r") as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    total = 0
    correct = 0
    mismatches = []

    for pred in predictions:
        idx = pred["index"]
        pred_answer = str(pred.get("answer", "") or "").strip().upper()
        gold_answer = correct_answers.get(idx, "").strip().upper()

        # Only count valid multiple-choice answers
        if gold_answer in ["A", "B", "C", "D", "E"]:
            total += 1
            if pred_answer == gold_answer:
                correct += 1
            else:
                mismatches.append({
                    "index": idx,
                    "predicted": pred_answer,
                    "correct": gold_answer
                })

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0
    accuracies[model_name] = (accuracy, correct, total)

    # Save mismatched predictions for analysis
    with open(os.path.join(metrics_dir, f"mismatches_{model_name.replace(' ', '_').lower()}.jsonl"), "w") as f:
        for entry in mismatches:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Prepare data for visualization
model_names = list(accuracies.keys())
accuracy_vals = [v[0] for v in accuracies.values()]
correct_vals = [v[1] for v in accuracies.values()]
total_vals = [v[2] for v in accuracies.values()]

# Plot bar chart of model accuracies
plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, accuracy_vals, color=["steelblue", "darkorange", "green"])
plt.title("Accuracy Comparison on GMAI-MMBench VAL Set")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)

# Annotate each bar with percentage and raw score
for bar, acc, corr, tot in zip(bars, accuracy_vals, correct_vals, total_vals):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
             f"{acc:.2%}\n({corr}/{tot})",
             ha='center', va='bottom')

plt.tight_layout()

# Save and display the plot
plot_path = os.path.join(metrics_dir, "accuracy_comparison.png")
plt.savefig(plot_path)
#plt.show()
