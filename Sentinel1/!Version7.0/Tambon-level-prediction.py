import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3.pkl")
#%%
for i, (column, dtype) in enumerate(zip(df.columns, df.dtypes.astype(str).tolist())):
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
df_dtypes = pd.concat([df.dtypes, df_compressed.dtypes], axis=1)
df_dtypes.columns = ["Before", "After"]


