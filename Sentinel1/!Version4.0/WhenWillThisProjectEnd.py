import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping, set_index_for_loc
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
strip_id = "402"
#%%
# Load df mapping
df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]

# Load df vew
df_vew = load_vew(root_df_vew, list_p)
df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
df_vew = pd.merge(df_vew, df_mapping, how="inner", on=["new_polygon_id"])

# load df s1_temporal (S1AB backscattering coef(s) starting from 2018-06-01 till the end of 2020)
df_s1_temporal = pd.concat(
    [pd.read_parquet(os.path.join(root_df_s1_temporal, file))
     for file in os.listdir(root_df_s1_temporal) 
     if file.split(".")[0][-3:] == strip_id
    ], 
    ignore_index=True
)
df_s1_temporal = df_s1_temporal.loc[df_s1_temporal.new_polygon_id.isin(df_mapping.new_polygon_id.unique())]
df_s1_temporal.columns = [column[:8] if "_S1" in column else column for column in df_s1_temporal.columns]
df_s1_temporal = set_index_for_loc(df_s1_temporal, column="new_polygon_id")

# Filter df_vew by final_plant_date (later or equal temporal first date)
df_vew = df_vew.loc[df_vew["final_plant_date"] >= datetime.datetime.strptime(df_s1_temporal.columns[7], "%Y%m%d")] # This is sentinal1(A|B) first date

# Create df look up table (for df_s1_temporal column)
df_content = pd.DataFrame(
    [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(7, len(df_s1_temporal.columns)-1)],
    columns=["index", "start", "stop"]
)
df_content = df_content.set_index("index")
df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
df_content = df_content.sort_index()
#%% Want to get only the data that within the crop cycle (plantdate to plantdate+180days)









