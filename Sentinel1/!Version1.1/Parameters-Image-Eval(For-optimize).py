# =============================================================================
# 1) ร้อยเอ็ด (p45)
# 2) ขอนแก่น (p40)
# 3) นครสวรรค์ (p60)
# 4) สุพรรณบุรี (p72)
# =============================================================================
import os
import sys
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
#from scipy import stats
#from skimage import filters
from general_fn import *
#import matplotlib.pyplot as plt
from osgeo import gdal, osr
import cv2
import rasterio
#from rasterio.mask import mask
import geopandas as gpd
from pierpy import io_tools

def load_dataframe_flood(root_df_flood):
    '''
    This function load all of the dataframe in root_df_flood folder then concat 
    where
        root_df_flood : where flood dataframe saved
    '''
    list_df = []
    for file in os.listdir(root_df_flood):
        df_flood = pd.read_parquet(os.path.join(root_df_flood, file))
        list_df.append(df_flood)
    df_flood = pd.concat(list_df, sort=False)
    return df_flood
    
def process_dataframe_flood(df_flood,
                            observed_index=None,
                            indices=None,
                            window_size=0):
    '''
    This function First, select only data in observed_index and selected indices
                  then find the minimum water value, minimum change value and maximum change value in observed window_size
    where
        df_flood : flood dataframe 
        observed_index : 
            - Default = None if want all of the dataframe
            - "PLANT_PROVINCE_CODE" or "strip_id" select observed index (province or strip_id) 
        indices : 
            - If observed_index=None, skip indices
            - list of observed index 
        'window_size' : size of observed window for min_water, max_change, min_change
    '''
    # Select only data in observed_index
    if observed_index!=None:
        df_flood = df_flood[df_flood[observed_index].isin(indices)]
    
    # Drop unnecessary columns
    df_flood = df_flood.drop(columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE'])
    
    # Select only data where loss_ratio>=0.8
    df_flood = df_flood[df_flood['loss_ratio']>=0.8]
    
    # Drop rows with Nan
    t_column_index = np.where(df_flood.columns == 't')[0][0]
    columns = df_flood.columns[t_column_index-window_size:t_column_index+window_size+1]    
    df_flood = df_flood.dropna(subset=columns)
    
    # Reset dataframe's index
    df_flood = df_flood.reset_index(drop=True)
    
    # Add min(flood) and min(flood)_date columns
    df_flood = df_flood.assign(min_water = df_flood[columns].min(axis=1),
                                         min_water_column = columns[np.argmin(df_flood[columns].values, axis=1)])
    
    # Add change value of each date (compare to previous date) (old-new)
    water_diff = -np.diff(df_flood[columns], axis=1)
    diff_columns = np.array([f"change_{columns[i]}_{columns[i+1]}" for i in range(len(columns)-1)])
    df_flood = df_flood.assign(**dict(zip(diff_columns, water_diff.T)))
    
    # Add max(change value) and max(change value) columns
    df_flood = df_flood.assign(max_change = df_flood[diff_columns].max(axis=1),
                                         max_change_column = diff_columns[np.argmax(df_flood[diff_columns].values, axis=1)],
                                         min_change = df_flood[diff_columns].min(axis=1),
                                         min_change_column = diff_columns[np.argmin(df_flood[diff_columns].values, axis=1)])
    return df_flood

def get_flood_parameters(df_flood,
                         water_percentile,
                         change_to_water_percentile,
                         change_from_water_percentile,
                         ):
    '''
    This function return value of selected percentile of water, change_to_water, change_to_land
    where
        df_flood : flood dataframe
        water_percentile : percentile for water threshold
        change_to_water_percentile : percentile for change to water threshold
        change_from_water_percentile : percentile for change from water threshold
    '''
    water_threshold =  df_flood['min_water'].quantile(water_percentile)
    change_to_water_threshold =  df_flood['max_change'].quantile(change_to_water_percentile)
    change_from_water_threshold =  df_flood['min_change'].quantile(change_from_water_percentile
                                          )
    return water_threshold, change_to_water_threshold, change_from_water_threshold

def save_parameters(threshold_data, filename_base, root_map_flood, raster_strip_id):
    '''
    This function save threshold parameters and details in .json file and return of saved path (for num_log)
    where
        threshold_data : dictionary of save data
        root_map_flood : save root 
    '''
    path_threshold_data = os.path.join(root_map_flood, f"{raster_strip_id}", f"{filename_base}_({threshold_data['water_percentile']:.2f},{threshold_data['change_to_water_percentile']:.2f},{threshold_data['change_from_water_percentile']:.2f}).json")
    if not os.path.exists(os.path.dirname(path_threshold_data)):
        os.mkdir(os.path.dirname(path_threshold_data))
    json.dump(threshold_data, open(path_threshold_data, 'w'))
    return path_threshold_data

def modeFilter(im, size):
    '''
    Mode filter function
    where
        'im' : input image
        'size' : filter size
    '''
    kernel = np.ones((size, size), 'float32')
    out = cv2.filter2D(im, ddepth=cv2.CV_8UC1, kernel=kernel)
    th = size**2-1
    th = th//2
    out = (out>th).astype('uint8')
    return out
    
def creat_flood_image(path_raster, threshold_data, selected_band = (0, 0), init=None):
    '''
    This function create flood image without saving raster into HDD
    where
        path_raster : path of selected strip_id raster
        threshold_data : dict of threshold {'raster_strip_id' : raster_strip_id,
                                            'water_threshold' : water_threshold,
                                            'change_to_water_threshold' : change_to_water_threshold,
                                            'change_from_water_threshold' : change_from_water_threshold}
        selected_band : create flood map of range(selected_band)
    '''
    raster = gdal.Open(path_raster)

    # Initialize parameters
    water_thresh = threshold_data['water_threshold']
    to_water_thresh =  threshold_data['change_to_water_threshold']
    to_land_thresh =  threshold_data['change_from_water_threshold']  
    
    if selected_band!=(0, 0):
        num_bands = selected_band[1] - selected_band[0] + 1 
        selected_band = (selected_band[0]-1, selected_band[1])
    else:
        num_bands = raster.RasterCount
    width = raster.RasterXSize
    height = raster.RasterYSize    
    valid_mask = np.ones((height, width), 'uint8')
    
    # Get valid_mask, and list of band_date
    if init is None:
        list_date = []
        for band in range(num_bands):
            raster_band = raster.GetRasterBand(selected_band[0]+band+1)
            band_name = raster_band.GetDescription()
            date = band_name[-8:]
            list_date.append(date)
            
            raster_im = raster_band.ReadAsArray()
            valid_mask *= (raster_im>0)
        
        row_valid, col_valid = np.nonzero(valid_mask)
        row_nvalid, col_nvalid = np.nonzero(valid_mask==0)
    else:
        list_date = init['list_date']
        row_valid, col_valid = init['valid']
        row_nvalid, col_nvalid = init['nvalid']
        
    # Create flood map
    map_flood = np.zeros((num_bands, height, width), np.uint8)
    im_flood = np.zeros((height, width), np.uint8)
    im_previous = None
    
    for band in range(num_bands):
        raster_band = raster.GetRasterBand(selected_band[0]+band+1)
        
        raster_im = raster_band.ReadAsArray()
        water_im = (raster_im<water_thresh).astype('uint8')
        
        if im_previous is not None:
            diff_im = im_previous - raster_im    
            changed_to_water_area = (diff_im>to_water_thresh).astype('uint8')
            
            # Pixel that change from land to water == 1 
            im_flood += changed_to_water_area
            changed_from_water_area = (diff_im<to_land_thresh).astype('uint8')
            
            # Find and remove flood's pixels that change from water to land and already land (from otsu)
            row_not_flood, col_not_flood = np.nonzero((changed_from_water_area)+(water_im==0)) # (change from water to land map + not water map))
            im_flood[row_not_flood, col_not_flood] = 0
            im_flood[row_nvalid, col_nvalid] = 0     
        
        # Save flood map and continue for next loop
        im_previous = raster_im.copy()
        
        map_flood[band] = (im_flood>0).astype('uint8')
#        map_flood[band] = modeFilter(im_flood, size=5)
    
    if init==None:
        list_date = [datetime.date(int(item[0:4]), int(item[4:6]), int(item[6:8])) for item in list_date]
        init = {'list_date' : list_date,
                'valid' : (row_valid, col_valid),
                'nvalid' : (row_nvalid, col_nvalid)}
    
    return map_flood, list_date, init

def find_date_index(list_raster_date, observed_date):
    '''
    This function return which index of sentinel's flood_date (axis=0)
    where
        list_raster_date : list of raster band date
        observed_date : flood_date or plant_date+90 days
    '''
    for idx, date in enumerate(list_raster_date): 
        if observed_date<date:
            break
    return idx

def get_sampling_date(gdf_selected):
    '''
    This function return which date to evaluate with flood map
    if no flood:
        return final_plant_date + 90 days
    if flood:
        return flood_date
    where
        gdf_selected : geodataframe of selected index
    '''
    # get date of normal case
    if pd.isna(gdf_selected['DANGER_TYPE_NAME']):
        date_plant = gdf_selected['final_plant_date'].date()

        # Add date offset for evaluate
        date_sampling = date_plant+datetime.timedelta(90)
    
    # get date of loss case
    else:
        date_sampling = gdf_selected['START_DATE'].date()
    return date_sampling

def load_dataframe_for_evaluation(path_df_mapping, path_polygon_activity, raster_strip_id):
    '''
    This function load mapping dataframe and polygon_activity dataframe and return only dataframe in selected raster_strip_id
    where
        path_df_mapping : path of mapping dataframe
        path_polygon_activity : path of all activities dataframe
        raster_strip_id : raster_strip_id to be evaluate
    '''
    '''
    Note 1) df_mapping is created from sentinel_pixel_detail
    '''
    df_mapping = pd.read_parquet(path_df_mapping)
    print("df_mapping loaded")
    gdf = io_tools.read_pickle(path_polygon_activity)
    print("gdf loaded")
    
    # Select only activities in selected strip_id
    df_mapping = df_mapping[df_mapping['strip_id'].isin([raster_strip_id])]
    gdf = gdf[gdf['polygon_id'].isin(df_mapping.index)]
     
    # Drop invalid final_plant_date
    gdf = gdf.dropna(subset=['final_plant_date'])
    
    # Select only Normal and Flood
    gdf = gdf[(gdf['DANGER_TYPE_NAME']=='อุทกภัย') | (pd.isnull(gdf['DANGER_TYPE_NAME']))]
    
    # Convert string into datetime
    gdf['final_plant_date'] = pd.to_datetime(gdf['final_plant_date'])
    gdf['START_DATE'] = pd.to_datetime(gdf['START_DATE'],  errors='coerce')
    
    # Drop case [flood and loss_ratio==0] :== select only [not flood or loss_ratio!=0]
    gdf = gdf[(gdf['loss_ratio']!=0) | pd.isna(gdf['START_DATE'])]
     
    # Drop out of range loss ratio
    gdf = gdf[(gdf['loss_ratio']==0) | (gdf['loss_ratio']>=0.8) & (gdf['loss_ratio']<=1)]

    gdf = gdf.reset_index(drop=True)    
    return df_mapping, gdf

def get_predict_and_ref(df_mapping, gdf, map_flood, raster_date):
    '''
    This function return predicted and ref loss ratio of every data in activity dataframe
    where
        df_mapping : mapping dataframe
        gdf : all activities dataframe
        path_raster_flood : path of flood map raster
    '''
    ref = []
    predicted = []
    for index in gdf.index:
        gdf_selected = gdf.loc[index]
        
        # Get sampling date
        sampling_date = get_sampling_date(gdf_selected)
        
        # Get band of satellite 
        satellite_index = find_date_index(raster_date, sampling_date) # index start from 0, band start from 1
    
        # Get row, col of satellite img
        rows = []
        cols = []
        for item in df_mapping.loc[gdf_selected['polygon_id'], 'row_col']:
            rows.append(int(item.split('_')[0]))
            cols.append(int(item.split('_')[1]))
         
        # Get Selected pixels 
    
        # Define range of sentinel_im index
        satellite_index_floor = max(0, satellite_index-flood_map_window_size)
        satellite_index_ceil = min(satellite_index+flood_map_window_size+1, len(map_flood))
        
        # Get selected pixel's values
        selected_pixels = map_flood[satellite_index_floor:satellite_index_ceil, rows, cols]  
        
        # Get predicted loss by operation_or of selected_pixels along axis=0
        predicted_loss = np.any(selected_pixels, axis=0).astype('uint8')
        
        # Get predicted_loss_ratio
        predicted_loss_ratio = predicted_loss.sum()/len(predicted_loss)
          
        # Get actual data
        ref_loss_ratio = gdf_selected['loss_ratio']
        
        # Add results into list
        predicted.append(predicted_loss_ratio)
        ref.append(ref_loss_ratio)

    return predicted, ref

def get_fuzzy_confusion_matrix(predicted, ref):
    '''
    This function return fuzzy confusion matrix
    where
        predicted is list of predicted loss ratio
        ref is list of groundtruth loss ratio
    '''
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

    # Add data in (Pred_flooded, Ref_flooded)
    f_cnf_matrix[0, 0] = df_results[['Pred_flooded', 'Ref_flooded']].min(axis=1).sum()
    # Add data in (Pred_flooded, Ref_no_flood)
    f_cnf_matrix[0, 1] = df_results[['Pred_flooded', 'Ref_no_flood']].min(axis=1).sum()
    # Add data in (Pred_no_flood, Ref_flooded)
    f_cnf_matrix[1, 0] = df_results[['Pred_no_flood', 'Ref_flooded']].min(axis=1).sum()
    # Add data in (Pred_no_flood, Ref_no_flood)
    f_cnf_matrix[1, 1] = df_results[['Pred_no_flood', 'Ref_no_flood']].min(axis=1).sum()

    return f_cnf_matrix
#%% Initial parameters
""" 
#p = 45 # ร้อยเอ็ด
#p = 60 # นครสวรรค์
#p = 72 # สุพรรณบุรี
"""
root_raster_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_flood"
root_map_flood = r"F:\CROP-PIER\CROP-WORK\Flood-Map"

path_polygon_activity = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster\polygon.pkl"
path_df_mapping = r"F:\CROP-PIER\CROP-WORK\Polygon-Pixel-Mapping\Polygon-Pixel-Mapping.parquet"

if not os.path.exists(root_map_flood):
    os.mkdir(root_map_flood)

# Inital variables of flood parameters >> For Image    
observed_index = 'strip_id'
indices = [108] 
window_size = 2
array_percentiles = np.arange(0.05, 1.00, 0.05)  # For n loop (Start, Stop, Step), For one loop (Start, Start+Step/2, Step)
change_from_water_percentile = 0.7

# Initial variables for Image
raster_strip_id = 108

selected_band = (0, 0)

# Initial variables for evaluation
flood_map_window_size = 2
#%% Create initial variable
filename_base = f"Flood_VV_S{raster_strip_id}"

#%% Load initial data (only once)
# Parameters
df_flood = load_dataframe_flood(root_df_flood)
df_flood = process_dataframe_flood(df_flood, observed_index, indices, window_size)

# Evaluation 
df_mapping, gdf = load_dataframe_for_evaluation(path_df_mapping, path_polygon_activity, raster_strip_id)
#%%
# =============================================================================
# START PROCESS (CAN START LOOP HERE)
# =============================================================================
#%%
# Create all combinaltion of selected values range
#threshold_parameters = np.stack(np.meshgrid(array_percentiles, array_percentiles)).T.reshape(-1, 2)[2::3]
threshold_parameters = np.stack(np.meshgrid(array_percentiles, array_percentiles)).T.reshape(-1, 2)
#%%
for water_percentile, change_to_water_percentile in tqdm(threshold_parameters):
    
# =============================================================================
#     Get flood parameters, and save parameters
# =============================================================================
    tqdm.write(f"Start({water_percentile:.2f}, {change_to_water_percentile:.2f}, {change_from_water_percentile:.2f})")
    water_threshold, change_to_water_threshold, change_from_water_threshold = get_flood_parameters(df_flood, water_percentile, change_to_water_percentile, change_from_water_percentile)
    
# =============================================================================
#     Create dictionary of threshold data
# =============================================================================
    threshold_data = {'raster_strip_id' : raster_strip_id,
                      'parameters_from' : indices,
                      'water_percentile' : water_percentile,
                      'change_to_water_percentile' : change_to_water_percentile,
                      'change_from_water_percentile' : change_from_water_percentile,
                      'water_threshold' : water_threshold,
                      'change_to_water_threshold' : change_to_water_threshold,
                      'change_from_water_threshold' : change_from_water_threshold
                     }
    
# =============================================================================
#     Save threshold parameters
# =============================================================================
    path_threshold_data = save_parameters(threshold_data, filename_base, root_map_flood, raster_strip_id)     
    if os.path.exists(os.path.join(root_map_flood, f'{raster_strip_id}', os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy')):
        continue                       
    tqdm.write(f"{path_threshold_data} Saved")
    
# =============================================================================
#     Create flood map
# =============================================================================
    path_raster = os.path.join(root_raster_sentinel1, [item for item in os.listdir(root_raster_sentinel1) if ('.' not in item) and (f'S{raster_strip_id}' in item)][0])
    if not 'init_flood' in locals():
        map_flood, list_date, init_flood = creat_flood_image(path_raster, threshold_data, selected_band = selected_band)

        # =============================================================================
        #     Clean gdf where start_date is too early (flood map can't cover at t-2)
        # =============================================================================
        
        gdf = gdf[~(gdf['START_DATE'] <= list_date[flood_map_window_size])]

    else:
        map_flood, list_date, _ = creat_flood_image(path_raster, threshold_data, selected_band = selected_band, init=init_flood)
# =============================================================================
#     Evaluation
# =============================================================================
    
    # Get predicted and ref loss ratio
    predicted, ref = get_predict_and_ref(df_mapping, gdf, map_flood, list_date)
    
    # Create Fuzzy Confusion Matrix
    f_cnf_matrix = get_fuzzy_confusion_matrix(predicted, ref)
    
    # Calculate OA, DT, FA
    OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
    DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall
    FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1])
    
    assert OA>=0
    assert DT>=0
    assert FA>=0
    
    results = { 
                'OA' : OA,
                'DT' : DT,
                'FA' : FA,  
                'total_flood' : float((gdf['loss_ratio']>0).sum()),
                'total_no_flood' : float((gdf['loss_ratio']==0).sum())
              }
    
    json.dump(results, open(os.path.join(root_map_flood, f"{raster_strip_id}", os.path.basename(path_threshold_data).split('.json')[0]+'_result.json'), 'w'))
    tqdm.write(f"{os.path.join(root_map_flood, str(raster_strip_id), os.path.basename(path_threshold_data).split('.json')[0]+'_result.json')} Saved")
    np.save(os.path.join(root_map_flood, f"{raster_strip_id}", os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy'), f_cnf_matrix) 
    tqdm.write(f"{os.path.join(root_map_flood, str(raster_strip_id), os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy')} Saved")
    tqdm.write(f"Detection rate: {results['DT']}")
    tqdm.write(f"False alarm rate: {results['FA']}")
    tqdm.write("")
#%%
