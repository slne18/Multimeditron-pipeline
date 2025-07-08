import pandas as pd
TSV_PATH = "/mloscratch/users/mberruye/evaluation/GMAI-MMBench/GMAI-MMBench_VAL.tsv"

df_gt = pd.read_csv(TSV_PATH, sep="\t")
print(df_gt.columns.tolist())