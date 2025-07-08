
# 🧠 MultiMeditron Pipeline

This repository contains code for evaluating and comparing multimodal language models (LLMs) on the GMAI-MMBench benchmark.

---

## 🔧 Setup

Make sure to activate the Python environment and install all required dependencies:

```bash
source venvmed/bin/activate
pip install -r requirements.txt
````

---

## 📦 Download Benchmark & Models (Required for Evaluation)

Because of size limitations, benchmark data and model checkpoints are **not included** in this repository. You must manually copy them from the cluster in the evaluation folder.

You need to download in the evaluation folder :

* The `GMAI-MMBench/` evaluation folder from:

  ```
  /mloscratch/users/nemo/benchmarking/GMAI-MMBench
  ```

* The `models/` folder (required **only** if you want to run Meditron inference):

  ```
  /mloscratch/users/nemo/benchmarking/models
  ```

> ⚠️ These folders are essential for the evaluation scripts to run correctly.

---

## 🚀 Evaluation Pipeline

### Step 1: Run the evaluation

Start by running the evaluation scripts to generate **cleaned answers** for each model. The output will be saved in the `evaluation/` folder under each model’s subdirectory.

Expected structure:

```
evaluation/
├── gemma/
│   └── cleaned_answers.jsonl
├── qwen/
│   └── cleaned_answers.jsonl
└── multimeditron/
    └── cleaned_answers.jsonl
```

### Step 2: Run GPT-based generation

Once all models have been evaluated and their outputs cleaned, you can proceed to run the **GPT-based judgment pipeline** via the `GPT_generation` module.

This step uses OpenAI's API to assess the quality of the model-generated answers based on multiple qualitative criteria.

Make sure the `evaluation/` folder is complete before starting this step.



