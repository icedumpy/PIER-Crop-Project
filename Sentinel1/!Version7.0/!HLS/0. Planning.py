import os
import pandas as pd
#%%
root = r"G:\!hls-prep\!hls-prep\ndvi-16d-noise-reduct-splitted"
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\!DRI-prep\pixel-values\df_dri_features_11_2017.parquet")
#%%
from icedumpy.geo_tools