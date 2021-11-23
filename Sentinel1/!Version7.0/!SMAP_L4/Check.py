import pandas as pd
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smsurface_temporal\SMAP_L4_smsurface_temporal_30_2020.parquet")
#%%
# df_sm = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\!SMAP_L4-prep\pixel-values\df_smsurface_features_96_2020.parquet")
df_sm = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\!SMAP_L4-prep\pixel-values\df_smsurface_features_30_2020.parquet")
df_sm = df_sm.drop(columns=["tambon_pcode", "amphur_cd", "tambon_cd"])

# Drop duplicates
df_sm = df_sm.reset_index()
df_sm = df_sm.drop_duplicates(subset=["row_col", "start_date", "sm_surface"])

# Find mean (day-level)
df_sm["start_date"] = df_sm["start_date"].str.slice(0, 10)
df_sm = df_sm.set_index("row_col")
#%%
date = "2020-10-03"
row = 148
col = 47
#%%
df_sm[df_sm["start_date"] == date].loc[f"{row}-{col}"].mean()
#%%
import os
import rasterio
import numpy as np
#%%
root = r"F:\CROP-PIER\CROP-WORK\!SMAP_L4-prep\raw\2020\sm_surface"
#%%
img = np.stack([rasterio.open(os.path.join(root, file)).read(1) for file in os.listdir(root) if "20200525" in file])
#%%
img[:, 54, 54].mean()
#%%