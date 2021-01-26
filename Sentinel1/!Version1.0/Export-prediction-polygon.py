import os
import sys
import json
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import datetime
import rasterio
import numpy as np
from tqdm import tqdm
from flood_fn import load_dataframe_mapping_vew, find_date_index, load_json, get_fuzzy_confusion_matrix, create_geodataframe
#%%
root_raster_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_flood_pixel"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
root_result = r"F:\CROP-PIER\CROP-WORK\Presentation\20200427"
#%%
# Initial variables for Image
raster_strip_id = 304
 
# Initial variables for evaluation
flood_map_window_size = 2

root_result = os.path.join(root_result, str(raster_strip_id))
if not os.path.exists(root_result):
    os.makedirs(root_result)
if not os.path.exists(os.path.join(root_result, 'result_polygon')):
    os.makedirs(os.path.join(root_result, 'result_polygon'))
#%%
path_threshold = r"F:\CROP-PIER\CROP-WORK\Flood-evaluation\\{0:d}\Flood_VV_S{1:d}_(0.80,0.20,0.70).json".format(raster_strip_id, raster_strip_id)
threshold_data = load_json(path_threshold)
df_mapping, df_vew = load_dataframe_mapping_vew(root_mapping, root_vew, root_raster_sentinel1, raster_strip_id, flood_map_window_size)
#%%
path_raster = os.path.join(root_raster_sentinel1, f"LSVVS{raster_strip_id}_2017-2019")
path_raster_flood = os.path.join(root_result, f"Flood_VV_S{raster_strip_id}.tiff")

raster = rasterio.open(path_raster)
raster_flood = rasterio.open(path_raster_flood)
raster_flood_im = raster_flood.read()
list_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
#%%  
ref = []
predicted = []   
for i in tqdm(range(len(df_vew))):
    df_vew_selected = df_vew.iloc[i]
    
    start_date = df_vew_selected['START_DATE']
    if start_date.year < 2017:
        break
    
    satellite_first_index = find_date_index(list_date, start_date-datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1
    satellite_last_index  = find_date_index(list_date, start_date+datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1

    satellite_first_index = max(0, satellite_first_index)
    satellite_last_index = min(satellite_last_index+1, len(raster_flood_im))  
    
    rows = []
    cols = []
    for item in df_mapping.at[df_vew_selected['new_polygon_id'], 'row_col']:
        rows.append(int(item.split('_')[0]))
        cols.append(int(item.split('_')[1]))     
    
    selected_pixels = raster_flood_im[satellite_first_index:satellite_last_index, rows, cols]  
    predicted_loss = np.any(selected_pixels, axis=0).astype('uint8')

    predicted_loss_ratio = predicted_loss.sum()/len(predicted_loss)
    ref_loss_ratio = df_vew_selected['loss_ratio']
 
    predicted.append(predicted_loss_ratio)
    ref.append(ref_loss_ratio)

f_cnf_matrix = get_fuzzy_confusion_matrix(predicted, ref)

# Calculate OA, DT, FA
OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall
FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1])
#%%
water_threshold = threshold_data['water_threshold']
to_water_threshold = threshold_data['change_to_water_threshold']
df_vew['predicted'] = predicted
df_vew['ref'] = ref
df_vew['result'] = np.nan
#%%
df_vew.loc[(df_vew['predicted'].astype('bool')==0) & (df_vew['ref'].astype('bool')==0), 'result'] = 'TN'
df_vew.loc[(df_vew['predicted'].astype('bool')==0) & (df_vew['ref'].astype('bool')==1), 'result'] = 'FN'
df_vew.loc[(df_vew['predicted'].astype('bool')==1) & (df_vew['ref'].astype('bool')==1), 'result'] = 'TP'
df_vew.loc[(df_vew['predicted'].astype('bool')==1) & (df_vew['ref'].astype('bool')==0), 'result'] = 'FP'
df_vew['date_result'] = df_vew['START_DATE'].dt.date.astype(str) + "_" + df_vew['result']
#%% Create polygon
gdf_vew = create_geodataframe(df_vew, polygon_column='final_polygon', crs={'init':'epsg:4326'})
gdf_vew['START_DATE_YEAR'] = gdf_vew['START_DATE'].dt.year
#%%
for year, gdf_vew_group in gdf_vew.groupby(['START_DATE_YEAR']):
    print(year, len(gdf_vew_group))   
    gdf_vew_group = gdf_vew_group[['new_polygon_id', 'final_plant_date', 'DANGER_TYPE_NAME', 'START_DATE', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA', 'TOTAL_DANGER_AREA_IN_WA', 'loss_ratio', 'predicted', 'result', 'date_result', 'geometry']]
    gdf_vew_group['final_plant_date'] = gdf_vew_group['final_plant_date'].astype(str)
    gdf_vew_group['START_DATE'] = gdf_vew_group['START_DATE'].astype(str)
    os.mkdir(os.path.join(root_result, 'result_polygon', str(year)))
    gdf_vew_group.to_file(os.path.join(root_result, 'result_polygon', str(year)), encoding = "CP874")
#%%
