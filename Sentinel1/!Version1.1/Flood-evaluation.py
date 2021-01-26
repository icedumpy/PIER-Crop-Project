import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pierpy import io_tools
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
from general_fn import *

def find_date_index(date_list, date):
    for idx, item in enumerate(date_list): 
        if date<item:
            break
    return idx

def get_sampling_date(gdf_selected):
    # get date of normal case
    if pd.isna(gdf_selected['DANGER_TYPE_NAME']):
        plant_date = gdf_selected['final_plant_date'].date()

        # Add date offset for evaluate
        sampling_date = plant_date+datetime.timedelta(90)
    
    # get date of loss case
    else:
        sampling_date = gdf_selected['START_DATE'].date()
    return sampling_date

def get_raster_pixel(satellite_index, flood_window_size, rows, cols):
    ''' Satellite_index is at flood_date
        In case of flood_window_size
            if flood_window_size = 0
                then satellite_index_floor, satellite_index_ceil has no effect (only predicted loss at flood_date selected return)
            else
                then return predicted_loss with selected window size
    '''
    
    satellite_index_floor = max(0, satellite_index-flood_window_size)
    satellite_index_ceil = min(satellite_index+flood_window_size+1, len(raster_im))
    
    # Get selected pixel's values
    selected_pixels = raster_im[satellite_index_floor:satellite_index_ceil, rows, cols]  
    return selected_pixels
#%% Define initial parameters
flood_window_size = 2
strip_id = 402
log_num = None
#%%
path_polygon = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster\polygon.pkl"
path_mapping = r"F:\CROP-PIER\CROP-WORK\Polygon-Pixel-Mapping\Polygon-Pixel-Mapping.parquet"
root_save = r"F:\CROP-PIER\CROP-WORK\Flood-Map"
#%%
_, path_log = load_log(root_save, strip_id, log_num=log_num)
path_save = os.path.join(root_save, os.path.basename(path_log).split('.')[0]+'_result.json')
#%%
path_raster_flood = os.path.join(root_save, os.path.basename(path_log).split('.')[0]+'.tiff')
print(path_raster_flood)
#raster_flood_path = r"D:\Crop\Data\GIS\Sentinel-1\Complete_VV\LSVVS205_2017-2019" # For check with original image
#%% Load df and preprocess
'''
Note 1) mapping_df is obtained from sentinel_pixel_detail
'''
mapping_df = pd.read_parquet(path_mapping)
print("mapping_df loaded")
gdf = io_tools.read_pickle(path_polygon)
print("gdf loaded")

# Select only activities in selected strip_id
mapping_df = mapping_df[mapping_df['strip_id'].isin([strip_id])]
gdf = gdf[gdf['polygon_id'].isin(mapping_df.index)]
 
# Drop invalid final_plant_date
gdf = gdf.dropna(subset=['final_plant_date'])

# Select only Normal and Flood
gdf = gdf[(gdf['DANGER_TYPE_NAME']=='อุทกภัย') | (pd.isnull(gdf['DANGER_TYPE_NAME']))]

# Convert string into datetime
gdf['final_plant_date'] = pd.to_datetime(gdf['final_plant_date'])
gdf['START_DATE'] = pd.to_datetime(gdf['START_DATE'],  errors='coerce')

# Drop case [flood and loss_ratio==0] :== select only [not flood or loss_ratio!=0]
gdf = gdf[(gdf['loss_ratio']!=0) | pd.isna(gdf['START_DATE'])]
gdf = gdf.reset_index(drop=True)
#%% Count n_sample
total_flood = gdf['DANGER_TYPE_NAME'].value_counts().to_dict()['อุทกภัย']
total_no_flood = len(gdf) - total_flood
#%% Load Raster 
raster = rasterio.open(path_raster_flood)
raster_im = raster.read()
raster_date = get_band_date(raster)
#%%
ref = []
predicted = []
for index in tqdm(gdf.index):
#    index = 27286 # Loss case
    gdf_selected = gdf.loc[index]
    
    # Get sampling date
    sampling_date = get_sampling_date(gdf_selected)
    
    # Get band of satellite 
    satellite_index = find_date_index(raster_date, sampling_date) # index start from 0, band start from 1

    # Get row, col of satellite img
    rows = []
    cols = []
    for item in mapping_df.loc[gdf_selected['polygon_id'], 'row_col']:
        rows.append(int(item.split('_')[0]))
        cols.append(int(item.split('_')[1]))
     
    # Get Selected pixels 
    try:
        # Try with selected flood_window_size
        selected_pixels = get_raster_pixel(raster_im, satellite_index=satellite_index, flood_window_size=flood_window_size, rows=rows, cols=cols)
    except:
        # If satellinte_index with flood_window_size out of range
            # Then flood_window_size = 0
        selected_pixels = get_raster_pixel(raster_im, satellite_index=satellite_index, flood_window_size=0, rows=rows, cols=cols)
    
    # Get predicted loss by operation_or of selected_pixels along axis=0
    predicted_loss = np.any(selected_pixels, axis=0).astype('uint8')
    
    # Get predicted_loss_ratio
    predicted_loss_ratio = predicted_loss.sum()/len(predicted_loss)
      
    # Get actual data
    ref_loss_ratio = gdf_selected['loss_ratio']
    
    # Add results into list
    predicted.append(predicted_loss_ratio)
    ref.append(ref_loss_ratio)
#%%
# Create Fuzzy Confusion Matrix
'''
                    Ref
               |Flooded|No Flood|
Pred    Flooded|_______|________|
     No Flooded|_______|________|
'''
f_cnf_matrix = np.zeros((2, 2), dtype=np.float64)
predicted = np.array(predicted, dtype=np.float32)
ref = np.array(ref, dtype=np.float32)
df_results = pd.DataFrame({'Pred_flooded' : predicted, 'Ref_flooded' : ref, 'Pred_no_flood' : 1-predicted, 'Ref_no_flood' : 1-ref})
#%%
# Add data in (Pred_flooded, Ref_flooded)
f_cnf_matrix[0, 0] = df_results[['Pred_flooded', 'Ref_flooded']].min(axis=1).sum()
# Add data in (Pred_flooded, Ref_no_flood)
f_cnf_matrix[0, 1] = df_results[['Pred_flooded', 'Ref_no_flood']].min(axis=1).sum()
# Add data in (Pred_no_flood, Ref_flooded)
f_cnf_matrix[1, 0] = df_results[['Pred_no_flood', 'Ref_flooded']].min(axis=1).sum()
# Add data in (Pred_no_flood, Ref_no_flood)
f_cnf_matrix[1, 1] = df_results[['Pred_no_flood', 'Ref_no_flood']].min(axis=1).sum()
#%%
OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall
FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1]) 
#%% Test
print(f"Overall accuracy: {OA}")
print(f"Detection rate: {DT}")
print(f"False alaem rate: {FA}")
#%% Save results
results = { 
            'total_flood' : total_flood,
            'total_no_flood' : total_no_flood,
            'OA' : OA,
            'DT' : DT,
            'FA' : FA
          }
json.dump(results, open(path_save, 'w'))
np.save(os.path.join(root_save, os.path.basename(path_log).split('.')[0]+'_result_fcnf.npy'), f_cnf_matrix)