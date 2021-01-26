import os
from osgeo import gdal
import numpy as np
import pandas as pd
import icedumpy
from tqdm import tqdm
#%%
root = r"F:\CROP-PIER\CROP-WORK\LS8"
#%% Get all file
df_file = pd.DataFrame(columns = ["file_template", "band", "filepath"])
for r, d, f in os.walk(root):
    for file in f:
        if file.endswith("B4_TOA.TIF") or file.endswith("B5_TOA.TIF"):
            df_file = df_file.append(pd.Series({"file_template" :  "_".join(file.split("_")[:7]),
                                                "band" : file.split("_")[7],
                                                "filepath" : os.path.join(r, file)
                                                }),
                                     ignore_index=True)
#%%
for template, df_file_grp in tqdm(df_file.groupby(['file_template'])):
    raster_red = gdal.Open(df_file_grp.loc[df_file_grp['band']=='B4', 'filepath'].iloc[0])
    raster_nir = gdal.Open(df_file_grp.loc[df_file_grp['band']=='B5', 'filepath'].iloc[0])
    
    assert raster_red.GetProjection()==raster_nir.GetProjection()
    assert raster_red.GetGeoTransform()==raster_nir.GetGeoTransform()
    
    path_save = os.path.join(os.path.dirname(df_file_grp.loc[df_file_grp['band']=='B4', 'filepath'].iloc[0]), os.path.basename(df_file_grp.loc[df_file_grp['band']=='B4', 'filepath'].iloc[0]).replace("B4", "NDVI"))
    
    
    red = raster_red.ReadAsArray().astype("float64")
    nir = raster_nir.ReadAsArray().astype("float64")
    
    ndvi = np.where((nir+red) == 0., 0, (nir-red)/(nir+red))
    
    icedumpy.geo_tools.create_tiff(path_save=path_save,
                                   im = ndvi,
                                   projection = raster_red.GetProjection(),
                                   geotransform = raster_red.GetGeoTransform(),
                                   drivername = "GTiff",
                                   nodata = None,
                                   channel_first=True,
                                   dtype=gdal.GDT_Float64
                                   )