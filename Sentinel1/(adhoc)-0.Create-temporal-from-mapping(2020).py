import os
import datetime
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_mapping, set_index_for_loc
#%%
sat_type = "S1AB"
root_raster = os.path.join(r"C:\Users\PongporC\Desktop\temp", sat_type.upper())
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_ext_act_id_rowcol_map_prov_scene_v5(at-False)_2020"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
#%%
list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
#%%
for strip_id in list_strip_id:
    print(strip_id)
    path_df_s1_temporal = os.path.join(root_df_s1_temporal, f"df_{sat_type.lower()}_pixel_s{strip_id}.parquet")
    # if os.path.exists(path_df_s1_temporal):
    #     continue
    
    # Load df mapping
    df_mapping, list_p = load_mapping(root_df_mapping, strip_id=strip_id)
    
    # Load df vew
    df_vew = pd.concat(
        [pd.read_parquet(os.path.join(root_df_vew, file)) 
         for file in os.listdir(root_df_vew) 
         if file.split(".")[0].split("_")[-1] in list_p
         ], 
        ignore_index=True
    )
    df_vew = df_vew.loc[df_vew["ext_act_id"].isin(df_mapping["ext_act_id"].unique())]
    
    # Get row, col
    row = df_mapping["row"].values
    col = df_mapping["col"].values
    
    # Load raster
    path_raster = os.path.join(root_raster, f"{sat_type}_{strip_id}.vrt")
    raster = rasterio.open(path_raster)
    
    # Load raster img
    img = raster.read([i+1 for i, band in enumerate(raster.descriptions) if (band[:4] in ["2020", "2021"]) and (datetime.datetime.strptime(band[:8], "%Y%m%d") > df_vew["final_plant_date"].min())])
    list_band = [band for i, band in enumerate(raster.descriptions) if (band[:4] in ["2020", "2021"]) and (datetime.datetime.strptime(band[:8], "%Y%m%d") > df_vew["final_plant_date"].min())]
    
    # Update data
    for i in range(img.shape[0]):
        df_mapping = df_mapping.assign(**{list_band[i]:img[i, row, col]})
        
    # Save as df temporal
    df_mapping.to_parquet(path_df_s1_temporal)
#%%