import os
import datetime
import numpy as np
from osgeo import gdal
# pip install git+https://github.com/icedumpy/icedumpy.git 
from icedumpy.geo_tools import create_tiff
#%%
# เอาไว้แยก raster ใหญ่ๆที่ stack กัน ให้ออกมาเป็นไฟล์แยกแต่ละ band อันนี้น่าจะเป็นของ gistda ที่เค้า stack มาให้
# รันสองรอบ S1A, S1B

# Folder เก็บ raster เก็บเอามาจาก gistda
root_complete_vv = r"G:\!PIER\!FROM_2TB\Complete_VV"

# เอาไว้เซฟ เพื่อใช้ต่อใน 1.2, 2, ...
root_s1a = r"G:\!PIER\!FROM_2TB\Complete_VV_separate\S1A"
os.makedirs(root_s1a, exist_ok=True)
#%%
for file in os.listdir(root_complete_vv):
    if not "." in file:
        path_src = os.path.join(root_complete_vv, file)
        raster = gdal.Open(path_src)
        strip_id = file.split("_")[0][-3:]
        os.makedirs(os.path.join(root_s1a, strip_id), exist_ok=True)
        
        projection = raster.GetProjection()
        geotransform = raster.GetGeoTransform()
        print(f"Start separting: {path_src}")
        for channel in range(1, raster.RasterCount+1):
            date = str(datetime.datetime.strptime(raster.GetRasterBand(channel).GetDescription()[-8:], "%Y%m%d").date())
            path_dst = os.path.join(root_s1a, strip_id, f"LSVVS{strip_id}_{date}.tif")
            print(f"Start channel: {channel}")
            if os.path.exists(path_dst):
                print(f"File {path_dst} already exists")
                print()
                continue
            
            raster_im = raster.GetRasterBand(channel).ReadAsArray()
            raster_im = np.expand_dims(raster_im, axis=0)
            
            create_tiff(path_save=path_dst,
                        im=raster_im,
                        projection=projection,
                        geotransform=geotransform,
                        drivername="GTiff",
                        nodata=None)
            print(f"Successfully saved: {path_dst}")
            print()