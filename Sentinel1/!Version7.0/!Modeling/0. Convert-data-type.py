import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3.pkl")
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
df.to_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
#%%
df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3.pkl")
df_compressed = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
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
