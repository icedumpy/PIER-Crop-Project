import os
import numpy as np
import pandas as pd
from icedumpy.geo_tools import convert_to_geodataframe
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_save = r"F:\CROP-PIER\CROP-WORK\vew_shp_sandbox"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if (file.split(".")[0][-7:-5] in ["30", "31", "36", "40"]) and (file.split(".")[0][-4:] in ["2019", "2020"])], ignore_index=True)
df = df[(df["DANGER_TYPE"].isna()) | (df["DANGER_TYPE"] == "อุทกภัย")]
df.loc[df["DANGER_TYPE"].isna(), "loss_ratio"] = 0
df.loc[df["DANGER_TYPE"] == "อุทกภัย", "loss_ratio"] = df.loc[df["DANGER_TYPE"] == "อุทกภัย", "TOTAL_DANGER_AREA_IN_WA"]/df.loc[df["DANGER_TYPE"] == "อุทกภัย", "TOTAL_ACTUAL_PLANT_AREA_IN_WA"]
df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]
df = df[df["in_season_rice_f"] == 1]
# df = df[(df["PLANT_AMPHUR_CODE"] == 9) & (df["PLANT_TAMBON_CODE"] == 3)]
df = df[['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'ext_act_id', 'TYPE_CODE', 'BREED_CODE',
         'TOTAL_ACTUAL_PLANT_AREA_IN_WA', 'TOTAL_DANGER_AREA_IN_WA', 'DANGER_DATE', 'DANGER_TYPE',
         'final_plant_year', 'final_plant_month', 'final_plant_date', 'final_harvest_date', 
         'final_crop_year', 'in_season_rice_f', 'loss_ratio', 'polygon',
         ]]
df.columns = ["p_code", "a_code", "t_code", "ext_act_id", "type_code", "breed_code",
              "pt_area_wa", "ls_area_wa", "ls_date", "ls_type",
              "pt_year", "pt_month", "pt_date", "harvt_date",
              "crop_year", "in_season", "loss_ratio", "polygon"]
df["pt_date"] = df["pt_date"].astype(str)
df["harvt_date"] = df["harvt_date"].astype(str)
df["ls_date"] = df["ls_date"].astype(str)
gdf = convert_to_geodataframe(df, polygon_column="polygon")
gdf = pd.concat([gdf[gdf["loss_ratio"] > 0], (gdf[gdf["loss_ratio"] == 0]).sample(n=len(gdf[gdf["loss_ratio"] > 0]))], ignore_index=True)
#%%
gdf.to_file(r"F:\CROP-PIER\CROP-WORK\vew_shp_sandbox\s304.shp", encoding="cp874")






