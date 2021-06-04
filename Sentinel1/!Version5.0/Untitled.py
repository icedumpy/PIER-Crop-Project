import os
import pandas as pd
from icedumpy.df_tools import load_vew
#%%
root_vew_new = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_vew_old = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
#%%
df_vew_new = pd.concat([pd.read_parquet(os.path.join(root_vew_new, file)) for file in os.listdir(root_vew_new) if file.split(".")[0].split("_")[-2][1:] == "45"], ignore_index=True)
df_vew_old = load_vew(root_vew_old, ["45"])
#%%
df_vew_new["ext_act_id"].isin(df_vew_old["ext_act_id"]).sum()
