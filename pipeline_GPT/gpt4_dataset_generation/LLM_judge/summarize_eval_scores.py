import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of models
models = [ "gemma", "qwen"]

criteria = [
    "accuracy",
    "completeness",
    "instruction_following",
    "communication",
    "context_awareness"
]

# Input/output paths
eval_base = "output_eval"
os.makedirs("metrics_eval", exist_ok=True)
csv_path = "metrics_eval/score_summary.csv"
plot_path = "metrics_eval/score_radar_plot.png"

# Collect average scores per model
summary_data = []

for model in models:
    path = os.path.join(eval_base, model, "final_eval.jsonl")
    print(f"Processing: {path}")

    scores_list = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            scores = obj.get("scores", {})
            if all(k in scores for k in criteria):
                scores_list.append(scores)

    if scores_list:
        df = pd.DataFrame(scores_list)
        mean_scores = df.mean().to_dict()
        mean_scores["model"] = model
        summary_data.append(mean_scores)
    else:
        print(f" No valid scores for {model}")

# Save average scores table
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.set_index("model")
summary_df.to_csv(csv_path)
print("\n Saved summary table to:", csv_path)

# Display summary in terminal
print("\n Average Evaluation Scores per Model:\n")
print(summary_df.round(2))

# Plot radar chart
labels = criteria
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model in summary_df.index:
    values = summary_df.loc[model, labels].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(180 / num_vars)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 5)
ax.set_title("Model Evaluation – HealthBench Criteria", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

plt.tight_layout()
plt.savefig(plot_path)
print(f"\nRadar plot saved to: {plot_path}")
