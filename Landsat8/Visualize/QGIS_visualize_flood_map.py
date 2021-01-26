import os
import numpy as np
import icedumpy
from osgeo import gdal
import rasterio

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%% Initialize parameters
root_df_ls8 = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_pixel_from_mapping_v2"
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_save = r"F:\CROP-PIER\CROP-WORK\Model_visualization\Flood_map\Landsat-8(no_cloud)"
root_model = r"F:\CROP-PIER\CROP-WORK\Model_visualization\model_129048_129049_130048_130049_B2_B3_B4_B5_cloud_0" 
bands = ['B2', 'B3', 'B4', 'B5']
pathrows = ["129048", "129049", "130048", "130049"]
#pathrows = ["130048", "130049", "129049"]

roc_params = icedumpy.io_tools.load_h5(os.path.join(root_model, "roc_params.h5"))
path_model = os.path.join(root_model, "model.joblib")
threshold = get_threshold_of_selected_fpr(roc_params['fpr_test'], roc_params['thresholds_test'], selected_fpr=0.3)

for pathrow in pathrows:
    root_raster = rf"G:\!PIER\!FROM_2TB\LS8_VRT\{pathrow}"
    path_mask = os.path.join(root_save, pathrow, f"{pathrow}.shp")
    
    path_raster = os.path.join(root_raster, f"ls8_{pathrow}_B1_TOA.vrt")
    raster = gdal.Open(path_raster)
    to_crs = rasterio.open(path_raster).crs.to_dict()
    
    # Creat mask
    if not os.path.exists(path_mask.replace(".shp", ".tif")):
        icedumpy.geo_tools.create_rice_mask(path_mask, root_vew, root_df_ls8, pathrow, raster, to_crs)

    # Create flood map
    path_mask = path_mask.replace(".shp", ".tif")
    icedumpy.geo_tools.create_flood_map(root_save, root_raster, path_mask, path_model, threshold, pathrow, bands, val_nodata=0, val_noflood=0)
#%% Initialize parameters
