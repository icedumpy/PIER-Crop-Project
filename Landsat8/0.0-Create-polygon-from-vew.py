import os
import pandas as pd
from icedumpy.geo_tools import convert_to_geodataframe
from tqdm import tqdm
#%%
root = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_save = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged_shp"
#%%
for file in tqdm(os.listdir(root)[0::4]):
    path_save = os.path.join(root_save, file.replace("parquet", "shp"))
    if os.path.exists(path_save):
        continue
    
    df = pd.read_parquet(os.path.join(root, file))
    
    gdf = convert_to_geodataframe(df)
    gdf = gdf[gdf['geometry'].type == "MultiPolygon"]
    gdf['START_DATE'] = gdf['START_DATE'].astype(str)
    gdf = gdf[['new_polygon_id', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA',
               'final_plant_date', 'START_DATE', 'DANGER_TYPE_NAME',
               'TOTAL_DANGER_AREA_IN_WA', 'geometry']]
    
    gdf.to_file(path_save)