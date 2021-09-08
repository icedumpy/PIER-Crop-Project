import os 
import pandas as pd
from sklearn.metrics import confusion_matrix
#%%
root = r"C:\Users\PongporC\Desktop\RCT"
df = pd.read_parquet(os.path.join(root, "Result.parquet"))
df["actual_label"] = (df["loss_ratio"] > 0).astype("uint8")
df["predict_label"] = (df["predict"] > 0).astype("uint8")
#%%
cnf_matrix = confusion_matrix(df["actual_label"], df["predict_label"])
#%%
FA = cnf_matrix[0, 1]/(cnf_matrix[0, 0]+cnf_matrix[0, 1])
