import os
import pandas as pd
from multiprocessing import Pool
from icedumpy.geo_tools import convert_to_geodataframe
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_shp = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202_shp"
#%%
def worker(file):
    path_file = os.path.join(root_df_vew, file)
    path_shp = os.path.join(root_df_shp, file.replace(".parquet", ".shp"))
    if not os.path.exists(path_shp):
        df_vew = pd.read_parquet(path_file)
        df_vew["final_plant_date"] = df_vew["final_plant_date"].astype(str)
        df_vew["final_harvest_date"] = df_vew["final_harvest_date"].astype(str)    
        df_vew["START_DATE"] = df_vew["START_DATE"].astype(str)    
        
        gdf = convert_to_geodataframe(df_vew)
        gdf = gdf[gdf['geometry'].type == "MultiPolygon"]

        gdf.to_file(path_shp)
#%%
if __name__ == '__main__':
    pool = Pool(os.cpu_count()-1) # Create a multiprocessing Pool
    pool.map(worker, os.listdir(root_df_vew))  # process data_inputs iterable with pool
#%%