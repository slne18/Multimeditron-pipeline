
# 🧼 Cleaning Pipeline for Medical Dataset

This folder contains scripts to **preprocess, clean, and organize JSONL datasets** used in medical multimodal tasks. These steps help ensure data quality, consistency, and proper structure before benchmarking.

⚠️ Make sure to modify the paths inside the different scripts where needed !

---

## 🔧 0. Setup

Before running any scripts, make sure to install the necessary dependencies:

```bash
pip install -r requirements_checks.txt
```

---

## 1️⃣ `jsonl_check.py` — **Check and clean duplicate image entries**

This script checks for **duplicate images** in your `.jsonl` dataset. It:

* Identifies and removes duplicate image entries based on file names or IDs
* Ensures consistent format (converts to `"conversations"` format if needed)
* Saves the cleaned result in a new folder: `clean_dataset/`
* If no duplicates are found, the dataset is still copied (unchanged) into the same folder for consistency

📌 **Usage**:

```bash
nohup python jsonl_check.py path/to/folder_or_file > jsonl_check.log 2>&1 &
```
---

## 2️⃣ `extract_modalities.py` — **Split dataset by image modality**

This script takes a `.jsonl` dataset and splits it into **separate files by modality type**.

✔️ Supported modalities include:
`MRI`, `CT`, `X-ray`, `Ultrasound`, `Microscopic`, `Photo`, etc.

* Entries with unknown or ambiguous modalities are grouped into:

  * `"General"` for uncommon types
  * `"Unknown"` if modality is missing or unrecognized

📁 Output: one `.jsonl` file per modality in the specified output folder.

📌 **Usage**:

```bash
nohup python extract_modalities.py path/to/input.jsonl path/to/output_folder > extract_modalities.log 2>&1 &
```

---

## 3️⃣ `similarity_analysis.py` *(Optional)* — **Analyze similarity of image captions**

This optional script checks for **caption redundancy** across datasets.

It:

* Calculates similarity scores between image captions
* Flags near-duplicates or highly similar captions
* Helps detect copy-pasted or repeated text descriptions

Helpful for auditing and improving dataset quality, though it is computationally intensive and may take considerable time to complete.

📌 **Usage**:

```bash
nohup python similarity_analysis.py path/to/folder_or_file > similarity_analysis.log 2>&1 &
```

---

## ✅ Summary

| Step | Script                   | Purpose                                 |
| ---- | ------------------------ | --------------------------------------- |
| 0    | `pip install ...`        | Install required packages               |
| 1    | `jsonl_check.py`         | Remove duplicates and unify format      |
| 2    | `extract_modalities.py`  | Separate files by modality type         |
| 3    | `similarity_analysis.py` | (Optional) Check for redundant captions |
