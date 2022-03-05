import os
import rasterio
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
#%%
# Read shapefile
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\shapefile\gdf_sel_plot_ks_plant_info.shp")
gdf = gdf[gdf["t_code"] == 460305]
# Read raster
raster = rasterio.open(r"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\01.tif")
#%%
for index, series in tqdm(gdf.iterrows(), total=len(gdf)):
    polygon = series.geometry
    arr, _ = mask(
        raster, 
        [polygon], 
        crop=True, 
        all_touched=False,
        indexes=1,
        nodata=-1
    )
    
    # Count unique of each cluster
    arr = arr[arr != -1]
    unique, counts = np.unique(arr, return_counts=True)
    dict_unique_count = dict(zip(unique, counts/len(arr)))
    
    # Assign portion of each class
    for i in range(20):
        if i in dict_unique_count.keys():
            gdf.loc[index, f"{i}"] = dict_unique_count[i]
        else:
            gdf.loc[index, f"{i}"] = 0
#%%
gdf = gdf.rename(columns={f"{i}":f"%cluster{i}"  for i in range(20)})
gdf.to_file(r"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\shapefile\cluster_portion.shp")
#%%
