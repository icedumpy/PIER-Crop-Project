import os
import pandas as pd
from sklearn.metrics import confusion_matrix
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
df_sandbox = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\Sandbox_ext_act_id.csv")
#%%
strip_id = "304"
df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df["in_season_rice_f"] == 1]
df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
#%%
df = df[df["ext_act_id"].isin(df_sandbox["ext_act_id"])]
#%%
