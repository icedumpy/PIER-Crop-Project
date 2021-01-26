import os
import rasterio 
import numpy as np
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from icedumpy.df_tools import set_index_for_loc
#%%
def gdal_rasterize(path_shp, path_rasterized,
                   xRes,
                   yRes,
                   outputBounds,
                   format="GTiff",
                   outputType="Byte",
                   burnValues=1,
                   allTouched=False,
                   noData=0):
    gdal_rasterize_cmd = f"gdal_rasterize -l {os.path.basename(path_shp).split('.')[0]}"
    gdal_rasterize_cmd += f" -burn {burnValues:f}"
    gdal_rasterize_cmd += f" -tr {xRes} {yRes}"
    gdal_rasterize_cmd += f" -a_nodata {noData}"
    gdal_rasterize_cmd += f" -te {' '.join(list(map(str, outputBounds)))}"
    gdal_rasterize_cmd += f" -ot {outputType}"
    gdal_rasterize_cmd += f" -of {format}"
    if allTouched:
        gdal_rasterize_cmd += " -at"
    gdal_rasterize_cmd += f' "{path_shp}" "{path_rasterized}"'
    result_merge_cmd = os.system(gdal_rasterize_cmd)
    assert result_merge_cmd == 0
    return gdal_rasterize_cmd
#%%
root_vew_shp = r"F:\CROP-PIER\CROP-WORK\Model_visualization\vew_shp"
root_rasterized = r"C:\Users\PongporC\Desktop\Drone Area\gistda_flood_2019\Rasterize"

path_gdf_gistda = r"C:\Users\PongporC\Desktop\Drone Area\gistda_flood_2019\flood_2019.shp"
path_gdf_thailand = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-province.shp"
path_shp = os.path.join(os.path.join(root_rasterized, "temp", "temp.shp"))

gdf_thailand = gpd.read_file(path_gdf_thailand)
gdf_thailand = set_index_for_loc(gdf_thailand, "ADM1_PCODE")
gdf_gistda = gpd.read_file(path_gdf_gistda)

reso = 10 # m
#%% Select "P"
for p in [int(p[1:]) for p in os.listdir(root_vew_shp)]:
    print(p)
    outputBounds = gdf_thailand.loc[f"TH{p}"].geometry.bounds
    
    try:
        # Rasterize vew
        vew_shp = pd.concat([gpd.read_file(os.path.join(root_vew_shp, f"p{p}", "start_date", file)) for file in os.listdir(os.path.join(root_vew_shp, f"p{p}", "start_date")) if file.endswith(".shp") and file[:4] == "2019"], ignore_index=True)
        vew_shp.to_file(path_shp)
        path_rasterized = os.path.join(root_rasterized, f"DOAE_p{p}.tif")
        gdal_rasterize(path_shp, path_rasterized, 
                       xRes=reso/111320, yRes=reso/111320,
                       outputBounds=outputBounds,
                       allTouched=True)
        
        # Rasterize gistda
        gdf_gistda_filtered = gdf_gistda.loc[gdf_gistda["PV_CODE"] == str(p)]
        if len(gdf_gistda_filtered) != 0:
            gdf_gistda_filtered.to_file(path_shp)
            path_rasterized = os.path.join(root_rasterized, f"GISTDA_p{p}.tif")
            gdal_rasterize(path_shp, path_rasterized, 
                           xRes=reso/111320, yRes=reso/111320,
                           outputBounds=outputBounds,
                           allTouched=True)
    except:
        pass
#%%