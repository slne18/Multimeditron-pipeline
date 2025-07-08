# Evaluation Scripts for Medical QCM Benchmark

This directory contains scripts to benchmark different models (MultiMeditron, Qwen, Gemma) on a medical multiple-choice questionnaire (QCM) benchmark. 

Raw outputs are saved in dedicated subfolders for each model. A post-processing script is also provided to clean the outputs by removing unwanted headers and unanswered questions.

## 1. benchmarking.py

### Description
This script runs the benchmark for a specified model and reasoning setting. Outputs are saved in:
outputs_benchmarks/model_name/


### Example nohup commands

#### MultiMeditron
```bash
nohup python benchmarking_multimeditron.py > logs/output_benchmarking_multimeditron.log 2>&1 &
nohup python benchmarking_gemma.py > logs/output_benchmarking_gemma.log 2>&1 &
nohup python benchmarking_qwen.py > logs/output_benchmarking_qwen.log 2>&1 &
```

## 2. postprocess_outputs.py

### Description
This script processes the raw outputs produced by the models. It:

Removes noisy headers or irrelevant text
Discards unanswered questions (e.g., empty responses or parsing failures)

#### Example command

```bash
nohup python postprocess_outputs.py > logs/output_postprocess.log 2>&1 &
```

