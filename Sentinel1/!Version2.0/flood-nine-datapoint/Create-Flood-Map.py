import os
import numpy as np
import flood_fn
import icedumpy

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%%
root_raster = r"G:\!PIER\!FROM_2TB\Complete_VV"
root_model = r"F:\CROP-PIER\CROP-WORK\Model_visualization\model_204_9"
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v3"

index_name = "s204"
model_name = index_name

root_flood_map = r"F:\CROP-PIER\CROP-WORK\Model_visualization\Flood_map\Sentinel-1"
os.makedirs(root_flood_map, exist_ok=True)
#%%
path_model = os.path.join(root_model, model_name+".joblib")
path_raster = os.path.join(root_raster, f"LSVV{index_name.upper()}_2017-2019")
path_mask = os.path.join(root_flood_map, index_name.replace('s', ''), 'rice_mask.shp')
path_save = os.path.join(root_flood_map, index_name.replace('s', ''), f"Flood_LSVV{index_name.upper()}_2017-2019_log_likelihood")
#%%s
model = flood_fn.load_model(path_model)
model.verbose = 0
model.n_jobs = -1
#%%
roc_params = icedumpy.io_tools.load_h5(os.path.join(root_model, "roc_params.h5"))
threshold = get_threshold_of_selected_fpr(roc_params['fpr_test'], roc_params['thresholds_test'], selected_fpr=0.3)
#%%
flood_im, raster_date =  flood_fn.create_flood_map(path_save, model, path_raster, path_mask, root_mapping, root_vew, index_name, threshold)