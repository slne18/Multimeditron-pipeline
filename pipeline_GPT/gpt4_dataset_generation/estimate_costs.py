import json
import os

# === CONFIGURATION ===
RESPONSES_DIR = "output_eval"  # Dossier avec les résultats de dataset_remaker
FILENAME = "final_eval.jsonl"  # Fichier de réponses récupérées
MODEL_PRICING = {
    "gpt-4o": {
        "prompt": 0.005,     # $ per 1K tokens
        "completion": 0.015  # $ per 1K tokens
    }
}
model_name = "gpt-4o"

# === LECTURE DES DONNÉES ===
total_prompt_tokens = 0
total_completion_tokens = 0
num_entries = 0

with open(os.path.join(RESPONSES_DIR, FILENAME), "r") as f:
    for line in f:
        entry = json.loads(line)
        try:
            usage = entry["full_response"]["response"]["body"]["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            num_entries += 1
        except Exception:
            continue

# === CALCUL DU COÛT ===
pricing = MODEL_PRICING[model_name]
cost_prompt = (total_prompt_tokens / 1000) * pricing["prompt"]
cost_completion = (total_completion_tokens / 1000) * pricing["completion"]
total_cost = cost_prompt + cost_completion

# === AFFICHAGE ===
print("=== ESTIMATION DES COÛTS ===")
print(f"Nombre total de requêtes : {num_entries}")
print(f"Prompt tokens utilisés : {total_prompt_tokens}")
print(f"Completion tokens utilisés : {total_completion_tokens}")
print(f"Coût prompt : ${cost_prompt:.4f}")
print(f"Coût completion : ${cost_completion:.4f}")
print(f"💰 Coût total estimé : ${total_cost:.4f}")
