import os
import pandas as pd
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
#%%
strip_id = "304"
df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
#%%
