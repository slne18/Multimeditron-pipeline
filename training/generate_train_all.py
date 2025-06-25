import os

log_folder = "/mloscratch/users/noize/processing-scripts/train_log"

# Define the base command format
command_template = (
    "nohup accelerate launch train_clip.py {config_file} --multi_gpu --num_processes 4 "
    "> {log_folder}/train_clip_{config_name}.log 2>&1"
)

# Path to your configuration files
config_folder = "/mloscratch/users/noize/processing-scripts/experts/configurations"

# List of your configuration file paths
config_files = [
   "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_aggressive_training_config_1.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_aggressive_training_config_2.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_aggressive_training_config_3.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_fine_tuning_config_1.yaml",  "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_fine_tuning_config_2.yaml",  "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_fine_tuning_config_3.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_initial_config_1.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_initial_config_2.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_initial_config_3.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_regularization_focused_config_1.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_regularization_focused_config_2.yaml", "/mloscratch/users/noize/processing-scripts/experts/configurations/combined_dataset_MRI_regularization_focused_config_3.yaml"
]

# Create the train_all.sh script with commands for sequential execution
with open('/mloscratch/users/noize/processing-scripts/experts/train_all.sh', 'w') as f:
    f.write("#!/bin/bash\n\n")
    
    for i, config_file in enumerate(config_files, 1):
        config_path = os.path.join(config_folder, config_file)
        log_name = f"train_clip_{i}"
        command = command_template.format(config_file=config_path, 
                                          log_folder=log_folder, 
                                          config_name=i)
        f.write(command + "\n")
        
print("train_all.sh script has been created with sequential execution.")