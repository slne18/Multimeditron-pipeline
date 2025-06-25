# 🧠 MRI Dataset Preprocessing Pipeline

This repository contains scripts for preprocessing and preparing **MRI datasets** from sources such as **MedTrinity** and **Radiopaedia**. The steps include format conversion, path correction, image extraction, dataset merging, cleaning, and splitting into train/test sets.

This pipeline can be adapted to any modality (such as CT, Ultrasound, etc.). 

⚠️ Make sure to modify the paths inside the different scripts where needed !

---

## 📦 0. Installation

Install the required dependencies:

```bash
pip install -r requirements_pipeline_MRI.txt
```
---

## 🔧 1. Convert MedTrinity MRI Data

`conversion_medtrinity.py`  
Preprocesses **MRI** data from MedTrinity stored in a JSONL file.

**What it does:**
- Cleans text content
- Updates image paths to a new directory structure
- Outputs a simplified, cleaned JSONL

✏️ **Don’t forget to edit:**
- Input JSONL path
- Output JSONL path
- `convert_path` method

**Run with:**

```bash
nohup python conversion_medtrinity.py > conversion_medtrinity.log 2>&1 &
```

---

## 🏥 2. Convert Radiopaedia MRI Data

`conversion_radiopedia.py`  
Processes **MRI** data from Radiopaedia, extracting reports and images into a clean JSONL format.

✏️ **Don’t forget to edit:**
- Input directory
- Output file path

**Run with:**

```bash
nohup python conversion_radiopedia.py > conversion_radiopedia.log 2>&1 &
```

---

## 🖼️ 3. Extract MRI Images

`extract_MRI_images.py`  
Copies MRI images listed in a JSONL file from a source directory to a target directory.

**What it does:**
- Copies only referenced images
- Logs missing ones

✏️ **Don’t forget to edit:**
- Input JSONL path
- Source and destination image folders

**Run with:**

```bash
nohup python extract_MRI_images.py > extract_MRI_images.log 2>&1 &
```

---

## 🔀 4. Combine Datasets

`combine_datasets.py`  
Combines the **Radiopaedia** and **MedTrinity** datasets into one JSONL file.

**What it does:**
- Groups Radiopaedia by patient
- Merges with MedTrinity data
- Outputs: `MRI_combined.jsonl`

✏️ **Don’t forget to edit:**
- Input paths for Radiopaedia and MedTrinity JSONLs
- Output path for combined file

**Run with:**

```bash
nohup python combine_datasets.py > combine_datasets.log 2>&1 &
```

---

## 🧹 5. Clean Invalid File Paths

`clean_lines.py`  
Removes entries from the dataset if any associated file paths do not exist on disk.

**What it does:**
- Verifies paths under the `modalities` key
- Outputs: `MRI_combined_clean.jsonl`

✏️ **Don’t forget to edit:**
- Input JSONL file: `MRI_combined.jsonl`
- Output cleaned file

**Run with:**

```bash
nohup python clean_lines.py > clean_lines.log 2>&1 &
```

---

## 🔂 6. Subsample Dataset

`remove_data.py`  
Samples a subset from a large dataset for quick experiments.

**What it does:**
- Shuffles and keeps the first N entries
- Outputs a reduced dataset

✏️ **Don’t forget to edit:**
- Input JSONL path
- Output JSONL path
- Number of examples to keep

**Run with:**

```bash
nohup python remove_data.py > remove_data.log 2>&1 &
```

---

## 📚 7. Split Dataset into Train/Test

`split_dataset_MRI.py`  
Splits the **MRI dataset** into training and test sets, ensuring patient-level separation.

**What it does:**
- 80/20 split by default (customizable)
- Output files: `MRI-train.jsonl`, `MRI-test.jsonl`

✏️ **Don’t forget to edit:**
- Input combined JSONL path
- Output paths for train/test JSONLs

**Run with:**

```bash
nohup python split_dataset_MRI.py > split_dataset_MRI.log 2>&1 &
```

### 🧠 Dataset Preprocessing Summary

| Step | Script                          | Purpose                                    | Output                              |
| ---- | ------------------------------- | ------------------------------------------ | ----------------------------------- |
| 0    | `requirements_pipeline_MRI.txt` | Install dependencies                       | —                                   |
| 1    | `conversion_medtrinity.py`      | Clean MedTrinity data and fix paths        | Cleaned MedTrinity JSONL            |
| 2    | `conversion_radiopedia.py`      | Extract & clean Radiopaedia reports/images | Cleaned Radiopaedia JSONL           |
| 3    | `extract_MRI_images.py`         | Copy MRI images listed in JSONL            | Folder with copied MRI images       |
| 4    | `combine_datasets.py`           | Merge MedTrinity and Radiopaedia datasets  | `MRI_combined.jsonl`                |
| 5    | `clean_lines.py`                | Remove entries with invalid image paths    | `MRI_combined_clean.jsonl`          |
| 6    | `remove_data.py`                | Subsample N entries for testing            | Subsampled JSONL                    |
| 7    | `split_dataset_MRI.py`          | Split into train/test (patient-level)      | `MRI-train.jsonl`, `MRI-test.jsonl` |

