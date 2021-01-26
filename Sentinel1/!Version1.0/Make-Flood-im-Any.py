import os
import datetime
import sys
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import numpy as np
from osgeo import gdal
from flood_fn import create_tiff, find_date_index
#%%
path_raster = r"F:\CROP-PIER\CROP-WORK\Presentation\20200427\304\Flood_VV_S304.tiff"
path_raster_any = os.path.splitext(path_raster)[0] + "_any" + os.path.splitext(path_raster)[1]
window = 2
#%%
raster = gdal.Open(path_raster)
raster_im = raster.ReadAsArray()
raster_im_any = np.zeros_like(raster_im)
list_band_name = [raster.GetRasterBand(i+1).GetDescription() for i in range(raster.RasterCount)]
list_date = [datetime.datetime(int(item[0:4]), int(item[4:6]), int(item[6:8])) for item in list_band_name]
#%%
for i in range(raster_im.shape[0]):
#    print(max(0, i-window), min(raster_im.shape[0]+1, i+window+1))  
   raster_date = list_date[i]
    
   satellite_first_index = find_date_index(list_date, raster_date-datetime.timedelta(12*window)) # index start from 0, band start from 1
   satellite_last_index  = find_date_index(list_date, raster_date+datetime.timedelta(12*window)) # index start from 0, band start from 1
   
   satellite_first_index = max(0, satellite_first_index)
   satellite_last_index = min(satellite_last_index+1, len(list_date))     
   
   raster_im_any[i] = np.any(raster_im[satellite_first_index:satellite_last_index], axis=0).astype('uint8')
#%% Save new image
projection = raster.GetProjection()
geotransform = raster.GetGeoTransform() 
create_tiff(path_raster_any, raster_im_any, projection, geotransform, list_band_name, dtype=gdal.GDT_Byte, channel_first=True)
