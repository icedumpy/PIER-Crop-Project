import os
import rasterio
import numpy as np
from icedumpy.df_tools import load_mapping
#%%
root_gistda = r"F:\CROP-PIER\CROP-WORK\GISTDA-Flood\Rasterized"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_vew_plant_info_official_polygon_disaster_all_rice_by_year_mapping(at-False)"
root_df_gistda_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_all_rice_by_year_pixel(at-False)"
#%%
for strip_id in np.unique([file.split(".")[0][-3:] for file in os.listdir(root_df_mapping)]):
    print(strip_id)
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
    list_p = df_mapping["p_code"].unique().tolist()
    
    # Extract flood index
    row = df_mapping["row"].values
    col = df_mapping["col"].values
    for file in [file for file in os.listdir(root_gistda) if file.split(".")[0][-3:] == strip_id]:
        img = rasterio.open(os.path.join(root_gistda, file)).read()[0]
        df_mapping = df_mapping.assign(**{file[:17]:img[row, col]})
        
    # Save file
    for p_code, df_mapping_grp in df_mapping.groupby("p_code"):
        df_mapping_grp.to_parquet(os.path.join(root_df_gistda_flood, f"df_gistda_flood_pixel_p{p_code}_s{strip_id}.parquet"))
#%%