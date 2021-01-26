import os
from os.path import join as path_join
import numpy as np
import pandas as pd
from osgeo import gdal
#%% Define function
def get_raster_minx_miny_maxx_maxy(raster):
    topleft_x, reso_x, _, topleft_y, _, reso_y = raster.GetGeoTransform()
    width  = raster.RasterXSize
    height = raster.RasterYSize
    
    minx = topleft_x
    maxx = topleft_x + width*reso_x
    miny = topleft_y + height*reso_y
    maxy = topleft_y
    
    return minx, miny, maxx, maxy
#%% Initialize parameters
root_sen1a = r"G:\!PIER\!FROM_2TB\Complete_VV"
list_file = [file for file in os.listdir(root_sen1a) if not "." in file]
#df_sen1a_template = pd.DataFrame(columns = ["strip_id", "minx", "miny", "maxx", "maxy"])
#%%
list_sen1a_template = []
for file in list_file:
    path_file = path_join(root_sen1a, file)
    strip_id = file[5:8]
    
    raster = gdal.Open(path_file)
    minx, miny, maxx, maxy = get_raster_minx_miny_maxx_maxy(raster)
    print(f"Strip_id({strip_id}):", minx, miny, maxx, maxy)
    
    list_sen1a_template.append(pd.Series({"strip_id":strip_id,
                                          "minx":minx,
                                          "miny":miny,
                                          "maxx":maxx,
                                          "maxy":maxy}))
df_sen1a_template = pd.concat(list_sen1a_template, axis=1).T
df_sen1a_template.to_parquet(r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\sen1a_strip_id_")
#%%

