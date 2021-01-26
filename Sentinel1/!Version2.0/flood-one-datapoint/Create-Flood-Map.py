import os
import flood_fn
#%%
root_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_model = r"F:\CROP-PIER\CROP-WORK\Model"
root_flood_map = r"F:\CROP-PIER\CROP-WORK\Flood-map"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"

index_name = "s304"
model_name = index_name
#%%
path_model = os.path.join(root_model, model_name+".joblib")
path_raster = os.path.join(root_raster, f"LSVV{index_name.upper()}_2017-2019")
path_mask = os.path.join(root_flood_map, index_name.replace('s', ''), 'rice_mask.shp')
path_save = os.path.join(root_flood_map, index_name.replace('s', ''), f"Flood_LSVV{index_name.upper()}_2017-2019_log_likelihood")
#%%
model = flood_fn.load_model(path_model)
model.verbose = 1
model.n_jobs = -1
#%%
flood_im, raster_date =  flood_fn.create_flood_map(path_save, model, path_raster, path_mask, root_mapping, root_vew, index_name)