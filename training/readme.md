# Training modality experts with CLIP

⚠️ Make sure to modify the paths inside the different scripts where needed !

## 0. Install the dependencies with `pip install -r requirements_experts.txt`. It is recommended to have a dedicated virtual environment for training.

## 1. `config_maker.py`: making training configurations for the model

Generate configuration files for the fine-tuning of CLIP. Register the datasets you need and their relative weights directly in the code of this script.

Usage: `python config_maker.py`

## 2. `train_clip.py`: fine-tuning CLIP with a given configuration file

Fine-tune CLIP with a given configuration file.

Usage: `nohup accelerate launch train_clip.py config_file.yaml --multi_gpu --num_processes 4 > train_clip.log 2>&1 &`

## 4. `generate_train_all.py`: Generates a shell script (train_all.sh) that runs multiple training configurations sequentially for the CLIP model. This allows you to train multiple configurations one after the other, ensuring each configuration is run in its own session with a unique log file for monitoring progress.

Usage: `python generate_train_all.py`

## 5. `train_all.sh`: Runs multiple training configurations sequentially for the CLIP model.
`nohup bash train_all.sh > train_all.log 2>&1 &`


