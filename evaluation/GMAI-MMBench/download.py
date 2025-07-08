from huggingface_hub import snapshot_download
import pandas as pd

REPO_ID = "OpenGVLab/GMAI-MMBench"
snapshot_download(repo_id=REPO_ID, local_dir="./", repo_type="dataset")
