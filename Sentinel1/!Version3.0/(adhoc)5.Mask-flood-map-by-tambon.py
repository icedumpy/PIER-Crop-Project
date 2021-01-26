import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import plot
from rasterio.mask import mask
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
#%% Define constant parameters
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp"
path_gdf_sent1_index = r"F:\CROP-PIER\COPY-FROM-PIER\Sentinel1-Index-our\Sentinel1_Index_4326_our.shp"
#%% Define parameters
sat_type = "S1AB"

if sat_type in ["S1B", "S1AB"]:
    list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
else:
    list_strip_id = ['101', '102', '103', '104', '105', '106', '107', '108', '109',
                     '201', '202', '203', '204', '205', '206', '207', '208',
                     '301', '302', '303', '304', '305', '306', 
                     '401', '402','403']
#%% Load tambon shapefile and process
gdf_tambon = gpd.read_file(path_gdf_tambon)
gdf_tambon["total_pixel"] = np.nan
gdf_sen1_index = gpd.read_file(path_gdf_sent1_index)
gdf_sen1_index = gdf_sen1_index[gdf_sen1_index["SID"].isin(list_strip_id)]
gdf_sen1_index = gdf_sen1_index.reset_index(drop=True)

df_tambon_within_sen1_index = gdf_sen1_index["geometry"].progress_apply(lambda val: gdf_tambon.intersects(val))
df_tambon_within_sen1_index = df_tambon_within_sen1_index.T
df_tambon_within_sen1_index.columns = list_strip_id
df_tambon_within_sen1_index = df_tambon_within_sen1_index.loc[df_tambon_within_sen1_index.sum(axis=1) > 0]
#%%
model_strip_id = "all"
for strip_id in list_strip_id:
    print(strip_id)
    # strip_id = "304"
    if model_strip_id == "all":
        root_raster = os.path.join(r"G:\!PIER\!Flood-map\Sentinel-1", sat_type, "Bins", f"{strip_id}-{model_strip_id}")
    else:
        root_raster = os.path.join(r"G:\!PIER\!Flood-map\Sentinel-1", sat_type, "Bins", strip_id)

    gdf_tambon_selected_strip_id = gdf_tambon.loc[df_tambon_within_sen1_index.loc[df_tambon_within_sen1_index[strip_id]].index]
    path_raster = os.path.join(root_raster, [file for file in os.listdir(root_raster) if file.endswith(".tif")][-4])
    raster = rasterio.open(path_raster)
    for row in tqdm(gdf_tambon_selected_strip_id.itertuples()):
        masked, affine_transform = mask(raster, [row.geometry],
                                    crop=True, indexes=1)
        
        total_pixel = (masked != 0).sum()
        if total_pixel == 0:
            continue
        percent_normal_confidence = (masked >= 2).sum()/total_pixel
        percent_high_confidence = (masked == 3).sum()/total_pixel
        
        # If no data or total_pixel higher than previous strip_id
        if np.isnan(gdf_tambon.loc[row.Index, "total_pixel"]) or total_pixel > gdf_tambon.loc[row.Index, "total_pixel"]:
            gdf_tambon.loc[row.Index, "total_pixel"] = total_pixel
            gdf_tambon.loc[row.Index, "percent_normal_confidence"] = percent_normal_confidence
            gdf_tambon.loc[row.Index, "percent_high_confidence"] = percent_high_confidence
#%%
gdf_tambon = gdf_tambon.loc[df_tambon_within_sen1_index.index]
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20201123\Tambon-score.shp")
#%%
# ax = gpd.GeoSeries(row.geometry).plot(facecolor="none", hatch=".", edgecolor="red",
#                                       alpha=0.5)
# plot.show(masked, transform=affine_transform, ax=ax, cmap="jet")
