import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icedumpy.df_tools import load_mapping, load_vew, clean_and_process_vew
#%% Visualize Loss ratio and Plot size
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
#%%
strip_id = "402"
#%%
df_mapping, list_p = load_mapping(root_df_mapping, strip_id=strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & (df_mapping["is_within"])]
df_mapping = df_mapping.drop_duplicates(subset=["new_polygon_id"])
df_vew = clean_and_process_vew(load_vew(root_df_vew, list_p), list_new_polygon_id=df_mapping["new_polygon_id"])
df_vew = df_vew.loc[df_vew["DANGER_TYPE_NAME"] == "อุทกภัย"]
df_vew = pd.merge(df_vew, df_mapping[["new_polygon_id", "polygon_area_in_square_m"]], how="inner", on=["new_polygon_id"])
df_vew["polygon_area_in_square_m"] = df_vew["polygon_area_in_square_m"]/2500
#%%
bins = 100
sns.displot(data=df_vew, x="loss_ratio", y="polygon_area_in_square_m", binwidth=(1/bins, df_vew["polygon_area_in_square_m"].max()/bins))
