import os
from osgeo import gdal
import icedumpy
from tqdm import tqdm

root_raster = r"F:\CROP-PIER\CROP-WORK\LS8_VRT"
root_save = r"F:\CROP-PIER\CROP-WORK\Model_visualization\LS8_color_infrared"
#%%
pathrows = ["129048", "129049", "130048", "130049"]
for pathrow in pathrows:
    tqdm.write(f"{pathrow}")
    root_raster_pathrow = os.path.join(root_raster, pathrow)
    raster = gdal.Open(os.path.join(root_raster_pathrow, os.listdir(root_raster_pathrow)[0]))
    projection = raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    drivername = "GTiff"    
    
    for channel in tqdm(range(1, raster.RasterCount+1)):
        path_save = os.path.join(root_save, pathrow, f"{raster.GetRasterBand(channel).GetDescription()}.tif")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        if os.path.exists(path_save):
            continue
        img = icedumpy.geo_tools.create_landsat8_color_infrared_img(root_raster=os.path.join(root_raster, pathrow), channel=channel, 
                                                                    rgb_or_bgr="rgb", channel_first=True)
        icedumpy.geo_tools.create_tiff(path_save=path_save,
                                       im = img,
                                       projection=projection,
                                       geotransform=geotransform,
                                       drivername=drivername,
                                       list_band_name=["B5", "B4", "B3"],
                                       channel_first=False)
