# Update df จากของ 3.1 จะได้ไม่ต้องรันใหม่หมด
import os
import rasterio
import pandas as pd
from osgeo import gdal
#%%
def create_trs(path_dst, path_src, file_format='ENVI'):
    outds = gdal.Translate(path_dst, path_src, format=file_format)
    outds.FlushCache()
    del outds
#%%
sat_type = "S1AB"
root_raster = os.path.join(r"C:\Users\PongporC\Desktop\temp", sat_type.upper())

# Folder ที่จะ update
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_pixel(at-False)"
#%%
for file_raster in [file for file in os.listdir(root_raster) if file.endswith(".vrt")]:
    strip_id = file_raster.split(".")[0][-3:]    
    
    # Load raster
    path_raster = os.path.join(root_raster, file_raster)
    raster = rasterio.open(path_raster)
    
    # Update df
    for file in [file for file in os.listdir(root_save) if file.split(".")[0][-3:] == strip_id]:
        print(f"\nLoading dataframe: {file}")
        df_mapping = pd.read_parquet(os.path.join(root_save, file))
        
        df_mapping_drop_dup = df_mapping.drop_duplicates(subset=["row", "col"])
        row = df_mapping_drop_dup["row"].values
        col = df_mapping_drop_dup["col"].values
        
        print("Updating pixel values")
        for i in range(2, raster.count):
            # print(raster.descriptions[i], raster.descriptions[i] in df_mapping_drop_dup.columns)
            if not raster.descriptions[i] in df_mapping_drop_dup.columns:
                img = raster.read(i+1)
                df_mapping_drop_dup = df_mapping_drop_dup.assign(**{raster.descriptions[i] : img[row, col]})
                
        #  Sort columns
        df_mapping_drop_dup = df_mapping_drop_dup[df_mapping_drop_dup.columns[:7].tolist()+sorted(df_mapping_drop_dup.columns[7:])]

        df_mapping = pd.merge(df_mapping.iloc[:, :7], 
                              df_mapping_drop_dup.drop(columns=["ext_act_id", "p_code", "polygon_area_in_square_m", "tier", "is_within"]),
                              how="inner", on=["row", "col"])
        del df_mapping_drop_dup
        
        print("Saving updated dataframe")
        df_mapping.to_parquet(os.path.join(root_save, file))
#%%



















