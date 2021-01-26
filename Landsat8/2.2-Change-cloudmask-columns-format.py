import os
import pandas as pd
root = r"F:\CROP-PIER\CROP-WORK\landsat8-dataframe\ls8-cloudmask"
#%%
for file in os.listdir(root):
    print(file)
    path_file = os.path.join(root, file)
    df = pd.read_parquet(path_file)
    df.rename(columns=lambda val: val.split("_")[-1][0:4] + "-" + val.split("_")[-1][4:6] + "-" + val.split("_")[-1][6:8]  if "cloudmask" in val else val, inplace=True)
    df.to_parquet(path_file)