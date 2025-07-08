# 🧠 MultiMeditron – Pipeline Overview

This repository is organized into **five main folders**, each representing a key step in the pipeline — from data cleaning to training and evaluation of multimodal models (e.g., CLIP variants).

⚠️ Make sure to modify the paths inside the different scripts where needed !

---

## 📁 1. `clean_extract_modalities/` — Data Cleaning & Modality Extraction

This is the **first folder to run**.
It contains scripts to:

* Clean raw `.jsonl` datasets (e.g., MedTrinity, Radiopaedia)
* Remove duplicates or formatting issues
* Automatically detect and **split data by imaging modality** (e.g., MRI, CT, X-ray)

➡️ Output: clean, modality-specific `.jsonl` files ready for preprocessing.

---

## 📁 2. `prepro_specific_modality/` — Preprocessing by Modality

This folder handles **preprocessing tailored to a specific imaging modality** (e.g., MRI, CT).

It includes:

* Format conversion
* Path correction
* Image extraction
* Dataset merging and train/test splitting

➡️ Output: structured and cleaned datasets, ready for training.

---

## 📁 3. `training/` — Contrastive Model Training (e.g., CLIP)

This folder contains training scripts for **vision-language models** on the processed datasets.

Models are trained to align image and text representations using contrastive learning (e.g., CLIP or MedCLIP-style training).

➡️ Output: trained model checkpoints.

---

## 📁 4. `evaluation_accuracy/` — Models' Accuracy and Precision Evaluation

This folder includes scripts to **evaluate the trained models** on downstream tasks such as:

* Visual question answering

It uses benchmark datasets like GMAI-MMBench and computes metrics such as **accuracy** and **precision**.

➡️ Output: metric reports and plots.

---

## 📁 5. `GPT_pipeline/` — GPT-Based Qualitative Evaluation

This folder handles **qualitative evaluation** of model responses using GPT-4 and includes two subfolders:

### 🔹 `evaluation/`
This is where you must place the outputs (`cleaned_answers.jsonl`) from the models after evaluation.  
Each model should have its own subfolder inside `evaluation/`:
GPT_pipeline/
└── evaluation/
├── gemma/
│ └── cleaned_answers.jsonl
├── qwen/
│ └── cleaned_answers.jsonl
└── multimeditron/
└── cleaned_answers.jsonl


### 🔹 `gpt_generation/`
This subfolder contains **three distinct pipelines** that interface with the OpenAI API:

1. **Answer Extraction**  
   Extract the predicted letter (A, B, C, D…) from free-form model outputs.

2. **Qualitative Scoring**  
   Use GPT-4 to score model answers on criteria like accuracy, completeness, instruction following, communication, and context awareness.

3. **Model Comparison**  
   Directly compare two model answers to the same question (pairwise judgment with preferred response and per-criterion scores).

---


## ✅ Summary

| Folder                      | Purpose                                                   | Run Order |
| --------------------------- | --------------------------------------------------------- | --------- |
| `clean_extract_modalities/` | Clean & split data by modality                            | 1         |
| `prepro_specific_modality/` | Preprocess data for a target modality                     | 2         |
| `training/`                 | Train CLIP multimodal model.                              | 3         |
| `evaluation_accuracy/`      | Evaluate models' accuracy and precision on benchmarks     | 4         |
| `GPT_pipeline/	            | Run GPT-based evaluation: extraction, scoring, comparison	| 5         |
