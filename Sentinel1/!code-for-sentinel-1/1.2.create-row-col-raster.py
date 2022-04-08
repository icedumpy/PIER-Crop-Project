import os
import numpy as np
from osgeo import gdal
# pip install git+https://github.com/icedumpy/icedumpy.git 
from icedumpy.geo_tools import create_tiff
#%%
# ที่แยกมาจาก 1.1
root_raster = r"G:\!PIER\!FROM_2TB\Complete_VV_separate"

# สร้าง row, col raster ใช้ต่อใน 1.3
root_raster_rowcol = r"G:\!PIER\!FROM_2TB\s1_pixel_rowcol_map"
#%%
for root, _, files in os.walk(root_raster):
    folder = os.path.basename(root)
    for file in files:
        print(os.path.basename(os.path.dirname(root)), folder, file)
        raster = gdal.Open(os.path.join(root, file))
        projection = raster.GetProjection()
        geotransform = raster.GetGeoTransform()
        row = np.expand_dims(np.arange(0, raster.RasterYSize).reshape(-1, 1).repeat(repeats=raster.RasterXSize, axis=1), axis=0)
        col = np.expand_dims(np.arange(0, raster.RasterXSize).reshape(-1, 1).T.repeat(repeats=raster.RasterYSize, axis=0), axis=0)
        
        path_row = os.path.join(root_raster_rowcol, f"s1_pixel_row_map_{folder}.tiff")
        path_col = os.path.join(root_raster_rowcol, f"s1_pixel_col_map_{folder}.tiff")
        
        create_tiff(path_save=path_row,
                    im=row,
                    projection=projection,
                    geotransform=geotransform,
                    drivername="GTiff",
                    nodata=None,
                    channel_first=True,
                    dtype="float32")
        
        create_tiff(path_save=path_col,
                    im=col,
                    projection=projection,
                    geotransform=geotransform,
                    drivername="GTiff",
                    nodata=None,
                    channel_first=True,
                    dtype="float32")
        break
