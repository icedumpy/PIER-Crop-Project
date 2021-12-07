import os
import rasterio
import pandas as pd
from osgeo import gdal
from icedumpy.df_tools import load_mapping
#%%
def create_trs(path_dst, path_src, file_format='ENVI'):
    outds = gdal.Translate(path_dst, path_src, format=file_format)
    outds.FlushCache()
    del outds
#%%
sat_type = "S1A"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_vew_plant_info_official_polygon_disaster_all_rice_by_year_mapping(at-False)"
root_raster = os.path.join(r"C:\Users\PongporC\Desktop\temp", sat_type.upper())
root_save = rf"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\{sat_type.lower()}_vew_plant_info_official_polygon_disaster_all_rice_by_year_pixel(at-False)"
os.makedirs(root_save, exist_ok=True)
#%%
for file_raster in [file for file in os.listdir(root_raster) if file.endswith(".vrt")]:
    strip_id = file_raster.split(".")[0][-3:]
    print(f"Start: S{strip_id}")
    # path_raster = os.path.join(root_raster, file_raster.replace(".vrt", ""))
    # if not os.path.exists(path_raster):
        # print("Creating raster in Drive:C")
        # create_trs(path_raster, path_raster+".vrt")
    path_raster = os.path.join(root_raster, file_raster)
    raster = rasterio.open(path_raster)
    list_p = [file.split(".")[0][-7:-5] for file in os.listdir(root_df_mapping) if file.split(".")[0][-3:] == strip_id]
    
    print("Loading df_mapping")
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[df_mapping["is_within"]]

    df_mapping_drop_dup = df_mapping.drop_duplicates(subset=["row", "col"])
    row = df_mapping_drop_dup["row"].values
    col = df_mapping_drop_dup["col"].values
    
    print("Extracting pixel values")
    for i in range(2, raster.count):
        print(raster.descriptions[i])
        img = raster.read(i+1)
        df_mapping_drop_dup = df_mapping_drop_dup.assign(**{raster.descriptions[i] : img[row, col]})
    
    df_mapping = pd.merge(df_mapping, 
                          df_mapping_drop_dup.drop(columns=["ext_act_id", "p_code", "polygon_area_in_square_m", "tier", "is_within"]),
                          how="inner", on=["row", "col"])
    
    del df_mapping_drop_dup
    
    print("Saving pixel values")
    for p, df_mapping_grp in df_mapping.groupby(["p_code"]):
        print(p, len(df_mapping_grp))
        df_mapping_grp.to_parquet(os.path.join(root_save, 
                                               f"df_{sat_type.lower()}_pixel_p{p}_s{strip_id}.parquet"))
    
    print(f"Finished: S{strip_id} desu yo")
    print()
#%%
















