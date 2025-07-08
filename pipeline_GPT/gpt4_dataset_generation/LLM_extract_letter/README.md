#  LLM Evaluation Pipeline – README

This pipeline supports the answer Extraction, using GPT to extract a letter (A–E) from model-generated answers.

---


## Full Pipeline Flow

This evaluation pipeline is designed to run after the main model benchmarking workflow. It follows the sequence below:

1. Benchmarking phase :
Run model inference over the GMAI-MMBench dataset.

2. Postprocessing phase : 
Clean and prepare model outputs into:
cleaned_answers.jsonl

3. LLM-based Extraction pipeline

* make_batches_extraction.py → Prepare GPT-based extraction prompts from reasoning and choices.

* make_api_queries.py → Submit batch prompts to the OpenAI API.

* dataset_remaker.py → Retrieve OpenAI responses and extract the final letter answer (A–E).

* merge_answers.py → Fill missing answer: null fields in cleaned_answers.jsonl using the extracted values.

4. Evaluation phase
compute_accuracy.py
→ Compute per-model accuracy and generate comparison plots.

This modular structure supports easy integration of additional models or datasets. To include a new model, simply ensure its cleaned_answers.jsonl and corresponding folder structure are in place—then update the model list in the evaluation scripts.

---


## Script Overview

### `make_batches_extraction.py`

**Purpose**: Prepares OpenAI batch requests to extract the **final answer letter** (`Answer: X`) from model explanations.

* Inputs:

  * `cleaned_answers.jsonl` (model outputs with potential `answer: null`)
  * `GMAI-MMBench_VAL.tsv` (provides choices A–E)
* Outputs: `.jsonl` requests in `batches_answer_extraction/`

**Run with:**

```bash
nohup python make_batches_extraction.py > logs/make_batches_extraction.log 2>&1 &
```

---

###  `make_api_queries.py`

**Purpose**: Submits all `.jsonl` batches to OpenAI’s API and stores the batch IDs.

* Reads: `batches_answer_extraction/`
* Saves batch IDs in: `output_answer_extraction/`

**Run with:**

```bash
nohup python make_api_queries.py > logs/make_api_queries.log 2>&1 &
```

---

###  `dataset_remaker.py`

**Purpose**: Retrieves OpenAI's responses and extracts the letter (`A`–`E`) from the model's answer.

* Reads batch results from: `output_answer_extraction/`
* Output: `final_extracted_answers.jsonl`

**Run with:**

```bash
nohup python dataset_remaker.py > logs/dataset_remaker.log 2>&1 &
```

---

###  `merge_answers.py`

**Purpose**: Fills missing `answer: null` fields in `cleaned_answers.jsonl` using GPT-extracted answers.

* Inputs:

  * `cleaned_answers.jsonl`
  * `final_extracted_answers.jsonl`
* Output: `merged_answers.jsonl`

**Run with:**

```bash
nohup python merge_answers.py > logs/merge_answers.log 2>&1 &
```

### `compute_accuracy.py`

**Purpose**: Computes accuracy scores by comparing predicted answers in `merged_answers.jsonl` against the ground truth from the TSV file.

* Inputs:
  * One `merged_answers.jsonl` file per model (e.g., `output_answer_extraction/gemma/merged_answers.jsonl`)
  * `GMAI-MMBench_VAL.tsv` (with correct answers)
* Output:
  * Accuracy scores printed to console
  * Bar chart saved as `metrics_eval/accuracy_comparison.png`
  * Mismatch logs saved as one `.jsonl` per model

This script runs **evaluation for three models** by default: `multimeditron`, `gemma`, and `qwen`.

**Run with:**

```bash
python compute_accuracy.py
```

---
