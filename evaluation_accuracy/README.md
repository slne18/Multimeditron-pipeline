# 🧪 Medical VQA Benchmark Pipeline

This repository evaluates the performance of multimodal medical models on the **GMAI-MMBench** benchmark, using both *reasoning* and *no-reasoning* approaches.

---

## 🔧 Setup

Make sure to activate the Python environment and install dependencies:

```bash
source venvmed/bin/activate
pip install -r requirements.txt
```

### 📦 Download Benchmark & Models (Required for Evaluation)

Because of size limitations, the benchmark and model files are **not included in this repository**.
You need to manually copy them from the cluster.

Specifically, you must download:

* The `GMAI-MMBench/` evaluation folder from: 

```bash
/mloscratch/users/nemo/benchmarking/GMAI-MMBench
```

* The `models/` folder (required **only if testing Meditron**) from:

```bash
/mloscratch/users/nemo/benchmarking/models
```

> ⚠️ These folders are essential for the evaluation scripts to run correctly.

---

## 📜 Pipeline Overview

The evaluation process follows these 4 main steps:

1. **Load the Model**

   * Models: `gemma-3-27b-it`, `MultiMedistron`, `Qwen2.5-VL-7B-Instruct`
   * All models support **image + text** inputs.

2. **Load the Benchmark**

   * Benchmark: `GMAI-MMBench`
   * Each sample contains:

     * A medical image
     * A question
     * A reference answer

3. **Generate Predictions** (with and without reasoning)

   * Input = (image, question)
   * Output = model-generated **free-form answer**
   * Extracted predicted letter (e.g., "A", "B", "C", "D", "E") using pattern recognition

4. **Score the Outputs**

   * Compare predictions to ground truth
   * Metrics used:

     * `Accuracy`: overall correctness
     * `Precision`: correctness when model commits to a class

---

## 📂 Scripts

⚠️ Make sure to modify the paths inside the different scripts where needed!

For the models `Gemma`, `MedGemma`, and `Qwen`, **two scripts** are available:

* One **with reasoning**
* One **without reasoning**

These scripts run the inference pipeline described above.

Their outputs are saved as `.jsonl` files in:

```
outputs_benchmarks/
```

Each line in the output includes:

* The original question
* The model's full answer
* The predicted letter (extracted by pattern matching)

---

## 🧹 Cleaning Outputs

To ensure consistent evaluation, run:

```bash
python clean_missing_entries.py
```

This will:

* Remove entries where the model fails to answer
* Remove answers where the letter could not be extracted

The result is a cleaned `.jsonl` file, saved in outputs_benchmarks/clean_missing_entries, and ready for scoring.

---

## 📈 Evaluation

Once cleaned outputs are ready, we compute and plot performance:

```bash
python compute_accuracy.py
```

This script:

* Computes accuracy and precision from the cleaned `.jsonl` files
* Combines them with **manually entered results** (e.g., for GPT-4o and MedGEMa)
* Saves plots to:

```
metrics_eval/
```

---

| Step | Description                                         | Output Folder                               | Key Scripts                           |
| ---- | --------------------------------------------------- | ------------------------------------------- | ------------------------------------- |
| 1    | **Generate predictions** with and without reasoning | `outputs_benchmarks/`                       | `benchmarking_<model>.py`             |
| 2    | **Clean missing/invalid entries**                   | `outputs_benchmarks/clean_missing_entries/` | `clean_missing_entries.py`            |
| 3    | **Compute accuracy & precision**                    | `metrics_eval/`                             | `compute_accuracy.py`                 |
| 4    | *(Optional)* Manually integrate external results    | `metrics_eval/`                             | — (done inside `compute_accuracy.py`) |

---

## 📌 Notes

* Accuracy is computed **only on answered questions** where the predicted letter could be extracted.
* This can under-represent models using reasoning, as long answers may prevent extraction.
* Ongoing work (by Mylène Berruyer) aims to improve robustness of the answer extraction and evaluation.


