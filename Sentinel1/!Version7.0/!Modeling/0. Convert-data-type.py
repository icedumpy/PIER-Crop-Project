import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
path = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_4a_NE3.parquet"
path_save = path.replace(".parquet", "_compressed.parquet")
#%%
df = pd.read_parquet(path)
#%%
for i, (column, dtype) in enumerate(zip(df.columns, df.dtypes.astype(str).tolist())):
    if column == "ext_act_id":
        continue
    print(f"{str(i).zfill(3)}: {column} { df[column].dtypes}")
    if (dtype.startswith("int")) and (df[column].min() >= 0) and (df[column].max() <= 255):
        df[column] = df[column].astype("uint8")
    elif dtype.endswith("64"):
        df[column] = df[column].astype(dtype.replace("64", "32"))
    print(f"{str(i).zfill(3)}: {column} { df[column].dtypes}")
    print()
#%%
df["y"] = df["y"].astype("uint8")
df.to_parquet(path_save)
#%%
df = pd.read_parquet(path)
df_compressed = pd.read_parquet(path_save)
#%%
dict_error = dict()
for column in df.columns:
    if not str(df[column].dtypes).startswith(("uint", "int", "float")):
        continue
    try:
        error = (df[column]-df_compressed[column]).mean()
        print(f"{column}: {error}")
        dict_error[column] = error
    except:
        print(f"{column}: Skip")
df_error = pd.DataFrame(dict_error.items())
#%%
print(df_error.diff(axis=1))