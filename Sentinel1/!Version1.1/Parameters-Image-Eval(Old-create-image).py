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

def save_parameters(threshold_data, filename_base, root_map_flood):
    '''
    This function save threshold parameters and details in .json file and return of saved path (for num_log)
    where
        threshold_data : dictionary of save data
        root_map_flood : save root 
    '''
    list_log = []
    for file in os.listdir(root_map_flood):
        if file.endswith('.json') and filename_base in file and 'result' not in file:
            list_log.append(file)   
    if len(list_log)==0:
        num_log = 1
    
    # Create next threshold_data file
    else:
        num_log = int(list_log[-1].split('.json')[0].split('_')[-1])+1
        
    path_threshold_data = os.path.join(root_map_flood, f"{filename_base}_{num_log}.json")
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

def creat_flood_raster(path_raster, filename_base, root_map_flood, num_log, threshold_data, video=False, selected_band = (0, 0)):
    '''
    This function create flood map
    where
        path_raster : path of selected strip_id raster
        root_map_flood: target path
        num_log : log number of saved file
        threshold_data : dict of threshold {'raster_strip_id' : raster_strip_id,
                                            'water_threshold' : water_threshold,
                                            'change_to_water_threshold' : change_to_water_threshold,
                                            'change_from_water_threshold' : change_from_water_threshold}
        video : want to save video or not
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
    
    # Create empty video file
    if video:
        frame_width = width//5
        frame_height = height//5
        out_video = cv2.VideoWriter(os.path.join(root_map_flood,  f"{filename_base}_{num_log}.avi"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (frame_width*2, frame_height))
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get valid_mask, and list of band_date
    date_list = []
    for band in range(num_bands):
        raster_band = raster.GetRasterBand(selected_band[0]+band+1)
        band_name = raster_band.GetDescription()
        date = band_name[-8:]
        date_list.append(date)
        
        raster_im = raster_band.ReadAsArray()
        valid_mask *= (raster_im>0)
    
    row_valid, col_valid = np.nonzero(valid_mask)
    row_nvalid, col_nvalid = np.nonzero(valid_mask==0)
    
    # Create empty raster
    driver = gdal.GetDriverByName('GTiff')
    out_image = driver.Create(os.path.join(root_map_flood, f"{filename_base}_{num_log}.tiff"), xsize=width, ysize=height, bands=num_bands, eType=gdal.GDT_Byte)
    src = osr.SpatialReference()
    src.ImportFromWkt(raster.GetProjection())
    out_image.SetProjection(src.ExportToWkt())
    out_image.SetGeoTransform(raster.GetGeoTransform())
    out_image.SetMetadata(raster.GetMetadata())
    
    # Create flood map
    flood_map = np.zeros((height, width), np.uint8)
    im_previous = None
    for band in range(num_bands):
        raster_band = raster.GetRasterBand(selected_band[0]+band+1)
        out_band = out_image.GetRasterBand(band+1)
        
        raster_im = raster_band.ReadAsArray()
        out_im = (raster_im<water_thresh).astype('uint8')
        
        if im_previous is not None:
            diff_im = im_previous - raster_im    
            changed_to_water_area = (diff_im>to_water_thresh).astype('uint8')
            
            # Pixel that change from land to water == 1 
            flood_map += changed_to_water_area
            changed_from_water_area = (diff_im<to_land_thresh).astype('uint8')
            
            # Find and remove flood's pixels that change from water to land and already land (from otsu)
            row_not_flood, col_not_flood = np.nonzero((changed_from_water_area)+(out_im==0)) # (change from water to land map + not water map))
            flood_map[row_not_flood, col_not_flood] = 0
            flood_map[row_nvalid, col_nvalid] = 0     
        
        # Save flood map and continue for next loop
        im_previous = raster_im.copy()
#        out_im = flood_map.astype('uint8')
        out_im = modeFilter(flood_map, size=5)
        out_band.WriteArray(out_im)
        out_band.SetDescription(date_list[band])
        out_band.FlushCache()
        
        # Write 1 video's frame
        if video:
            out_im *= 255   
            
            out_im[row_nvalid, col_nvalid] = 0
            
            out_im2 = cv2.resize(out_im, (frame_width, frame_height))
            out_im3 = np.zeros((frame_height, frame_width, 3), 'uint8')
            
            raster_im_resize = cv2.resize(raster_im, (frame_width, frame_height))
            vmin = np.percentile(raster_im_resize, 2)
            vmax = np.percentile(raster_im_resize, 98)
            
            raster_im_resize -= vmin
            raster_im_resize /= vmax
            
            raster_im_resize = 255.0*(raster_im_resize>1.0) + 255.0*raster_im_resize*(raster_im_resize<=1)*(raster_im_resize>=0)
            raster_im_resize_jet = cv2.applyColorMap(raster_im_resize.astype('uint8'), cv2.COLORMAP_JET)
            
            out_im3[:, :, 0] = out_im2
            out_im3[:, :, 1] = out_im2
            out_im3[:, :, 2] = out_im2
            cv2.putText(out_im3, date_list[band] + "_{}".format(band+1), (10, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("out_th", out_im3)
            cv2.imshow("raster_im_resize_jet", raster_im_resize_jet)
            cv2.waitKey(10)
            out_im3 = np.concatenate((out_im3, raster_im_resize_jet), axis=1)
            out_video.write(out_im3)
    if video:
        out_video.release()        
    out_image = None
    cv2.destroyAllWindows()

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
    gdf = gdf.reset_index(drop=True)
    return df_mapping, gdf

def get_predict_and_ref(df_mapping, gdf, path_raster_flood):
    '''
    This function return predicted and ref loss ratio of every data in activity dataframe
    where
        df_mapping : mapping dataframe
        gdf : all activities dataframe
        path_raster_flood : path of flood map raster
    '''
    raster = rasterio.open(path_raster_flood)
    raster_im = raster.read()
    
    # Get raster band date
    raster_date = get_band_date(raster)
    
    ref = []
    predicted = []
    for index in gdf.index:
    #    index = 27286 # Loss case
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
        satellite_index_ceil = min(satellite_index+flood_map_window_size+1, len(raster_im))
        
        # Get selected pixel's values
        selected_pixels = raster_im[satellite_index_floor:satellite_index_ceil, rows, cols]  
    
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
indices = [402] 
window_size = 2
array_percentiles = np.arange(0.7, 0.75, 0.1)  # For n loop (Start, Stop, Step), For one loop (Start, Start+Step/2, Step)
total_parameters = 3

# Initial variables for Image
raster_strip_id = 402
video = True
selected_band = (0, 0)

# Initial variables for evaluation
flood_map_window_size = 2

#%% Create initial variable
filename_base = f"Flood_VV_S{raster_strip_id}"

#%% Load initial data (only once)
# Parrameters
df_flood = load_dataframe_flood(root_df_flood)
df_flood = process_dataframe_flood(df_flood, observed_index, indices, window_size)

# Evaluation 
df_mapping, gdf = load_dataframe_for_evaluation(path_df_mapping, path_polygon_activity, raster_strip_id)
#%%
# =============================================================================
# START PROCESS (CAN START LOOP HERE)
# =============================================================================
'''
Desciption
    np.repeat((a, b, c), 2) >> (a, a, b, b, c, c)
    np.tile((a, b, c), 2) >> (a, b, c, a, b, c)
'''
# For outter loop(a, a, a, a, b, b, b, b)
water_percentiles = np.tile(np.repeat(array_percentiles, np.power(len(array_percentiles), total_parameters-1)), np.power(len(array_percentiles), total_parameters-3))

# For middle loop(a, a, b, b, a, a, b, b)
change_to_water_percentiles = np.tile(np.repeat(array_percentiles, np.power(len(array_percentiles), total_parameters-2)), np.power(len(array_percentiles), total_parameters-2))

# For Inner loop (a, b, a, b, a, b, a, b)
change_from_water_percentiles = np.tile(np.repeat(array_percentiles, np.power(len(array_percentiles), total_parameters-3)), np.power(len(array_percentiles), total_parameters-1))

for water_percentile, change_to_water_percentile, change_from_water_percentile in tqdm(zip(water_percentiles, change_to_water_percentiles, change_from_water_percentiles), total=len(water_percentiles)):
    #%% Get flood parameters, and save parameters
#    change_to_water_percentile = 1-change_to_water_percentile
    tqdm.write(f"Start({water_percentile}, {change_to_water_percentile}, {change_from_water_percentile})")
    water_threshold, change_to_water_threshold, change_from_water_threshold = get_flood_parameters(df_flood, water_percentile, change_to_water_percentile, change_from_water_percentile)
    
    # Create dictionary of threshold data
    threshold_data = {'raster_strip_id' : raster_strip_id,
                      'parameters_from' : indices,
                      'water_percentile' : water_percentile,
                      'change_to_water_percentile' : change_to_water_percentile,
                      'change_from_water_percentile' : change_from_water_percentile,
                      'water_threshold' : water_threshold,
                      'change_to_water_threshold' : change_to_water_threshold,
                      'change_from_water_threshold' : change_from_water_threshold
                     }
    # Save parameters
    path_threshold_data = save_parameters(threshold_data, filename_base, root_map_flood)
    #%% Create flood map
    path_raster = os.path.join(root_raster_sentinel1, [item for item in os.listdir(root_raster_sentinel1) if ('.' not in item) and (f'S{raster_strip_id}' in item)][0])
    num_log = int(path_threshold_data.split('.')[0].split('_')[-1])
    creat_flood_raster(path_raster, filename_base, root_map_flood, num_log, threshold_data, video=video, selected_band = selected_band)
    
    #%% Evaluation
    # Count n_sample
    total_flood = gdf['DANGER_TYPE_NAME'].value_counts().to_dict()['อุทกภัย']
    total_no_flood = len(gdf) - total_flood
    
    # Load raster and raster image
    path_raster_flood = os.path.join(root_map_flood, f"{filename_base}_{num_log}.tiff")
    
    # Get predicted and ref loss ratio
    predicted, ref = get_predict_and_ref(df_mapping, gdf, path_raster_flood)
    
    # Create Fuzzy Confusion Matrix
    f_cnf_matrix = get_fuzzy_confusion_matrix(predicted, ref)
    
    # Calculate OA, DT, FA
    OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
    DT = 100*f_cnf_matrix[0, 0]/total_flood # aka recall
    FA = 100*f_cnf_matrix[0, 1]/total_no_flood
    
    results = { 
                'total_flood' : total_flood,
                'total_no_flood' : total_no_flood,
                'OA' : OA,
                'DT' : DT,
                'FA' : FA
              }
    json.dump(results, open(os.path.join(root_map_flood, os.path.basename(path_threshold_data).split('.')[0]+'_result.json'), 'w'))
    np.save(os.path.join(root_map_flood, os.path.basename(path_threshold_data).split('.')[0]+'_result_fcnf.npy'), f_cnf_matrix) 
#%%
