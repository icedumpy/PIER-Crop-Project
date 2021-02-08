import os
import rasterio
import pandas as pd
from tqdm import tqdm
#%%
sat_type = "S1AB"
root_raster = os.path.join(r"C:\Users\PongporC\Desktop\temp", sat_type.upper())
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_ext_act_id_rowcol_map_prov_scene_v5(at-False)_2020"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
os.makedirs(root_save, exist_ok=True)
#%%
# Get list of files (sort by strip_id)
list_file = os.listdir(root_save)
list_file = sorted(list_file, key=lambda val: val.split(".")[0][-3:])
#%%
# Load for each file (Keep it simply but slowvy. For the sake of PC's memories)
pbar = tqdm(list_file)
for file in pbar:
    pbar.set_description(f"Processing: {file}")
    # Load raster
    p = file.split(".")[0].split("_")[-2][1:]
    strip_id = file.split(".")[0][-3:]
    path_raster = os.path.join(root_raster, f"{sat_type}_{strip_id}.vrt")
    raster = rasterio.open(path_raster)
    
    # Load df 
    path_file = os.path.join(root_save, file)
    path_df_mapping = os.path.join(root_df_mapping, f"df_s1_ext_act_id_rowcol_map_p{p}_s{strip_id}.parquet")
    
    df = pd.read_parquet(path_file)
    df_mapping = pd.read_parquet(path_df_mapping)
    
    # Keep only new row col
    df_mapping = df_mapping.loc[~(df_mapping["row"].astype(str) + "_" + df_mapping["col"].astype(str)).isin(df["row"].astype(str) + "_" + df["col"].astype(str))]
    
    # Change ext_act_id to new_polygon_id (wrong but however, I want only row, col)
    df_mapping = df_mapping.rename(columns={"ext_act_id":"new_polygon_id"})
    
    # Update basic info to df
    df = pd.concat([df, df_mapping], ignore_index=True)
    
    # Get row, col
    row = df_mapping["row"].values
    col = df_mapping["col"].values

    # Loop for each raster's band
    for i in range(2, raster.count):
        print(raster.descriptions[i])
        img = raster.read(i+1)
        df.loc[len(df)-len(df_mapping):, raster.descriptions[i]] = img[row, col]
    
    # Save updated df
    df.to_parquet(path_file)
#%%
