import os
import re
import json
import datetime
from joblib import dump, load
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from osgeo import gdal
import rasterio

def save_model(path, model):
    dump(model, path) 
    
def load_model(path):
    return load(path) 

def load_json(path):
    return json.load(open(path, 'r'))

def get_band_date(raster):
    ''' Extract band date from band name'''
    return [datetime.date(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]

def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def create_geodataframe(df, polygon_column='final_polygon', crs={'init':'epsg:4326'}):
    gdf = gpd.GeoDataFrame(df)
    gdf['geometry'] = gdf[polygon_column].apply(wkt.loads)
    gdf.crs = crs
    return gdf

def create_tiff(path_save, im, projection, geotransform, list_band_name, dtype=gdal.GDT_Byte, channel_first=True):
    if len(im.shape)==2:
        im = np.expand_dims(im, 0)
    if not channel_first:
        im = np.moveaxis(im, 0, -1)
    
    band = im.shape[0]
    row = im.shape[1]
    col = im.shape[2]
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(path_save, col, row, band, dtype)
    output.SetProjection(projection)
    output.SetGeoTransform(geotransform)
    for i in range(band):
        band_name = list_band_name[i]
        print(band_name)
        output.GetRasterBand(i+1).SetDescription(band_name)
        output.GetRasterBand(i+1).WriteArray(im[i, :, :])   
        output.FlushCache()
    del output
    del driver

def load_dataframe_flood(root_df_flood, index_name):
    '''
    Load flood dataframe of selected index : 'pxx' or 'sxxx' where is p_code or strip_id
    '''
    
    # Load and concat all dataframes with the same strip_id
    df_flood = pd.concat([pd.read_parquet(os.path.join(root_df_flood, file)) for file in os.listdir(root_df_flood) if index_name in file], ignore_index=True)
    
    # Get the index of 't' column
    t_column_index = np.where(df_flood.columns == 't')[0][0]
    
    # get ['t-2', 't-1', 't', 't+1', 't+2']
    columns = df_flood.columns[t_column_index-2:t_column_index+2+1]
    
    # Drop all 0 data
    df_flood = df_flood[~np.all(df_flood[columns]==0, axis=1)]
    
    # Drop nan(any) data
    df_flood = df_flood[~np.any(pd.isna(df_flood[columns]), axis=1)]
    
    df_flood = df_flood.dropna(subset=columns)
    df_flood.drop(columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE'], inplace=True)
    df_flood = df_flood[(df_flood['loss_ratio']>=0.8) & (df_flood['loss_ratio']<=1.0)]
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

def load_dataframe_mapping_vew(root_mapping, root_vew, root_raster_sentinel1, raster_strip_id, flood_map_window_size):
    '''
    Load mapping, vew dataframe of selected raster_strip_id
    '''
    df_mapping = pd.concat([pd.read_parquet(os.path.join(root_mapping, file)) for file in os.listdir(root_mapping) if f's{raster_strip_id}' in file]) 
    list_p_code = [file[28:31] for file in os.listdir(root_mapping) if f's{raster_strip_id}' in file]
    
    df_vew = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if os.path.splitext(file)[0][-3:] in list_p_code], ignore_index=True)
    df_vew = df_vew[df_vew['new_polygon_id'].isin(df_mapping.index)]
    df_vew['loss_ratio'] = np.where(np.isnan(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']), 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA'])
    
    # Drop invalid final_plant_date
    df_vew = df_vew.dropna(subset=['final_plant_date'])
    
    # Select only Normal and Flood
    df_vew = df_vew[(df_vew['DANGER_TYPE_NAME']=='อุทกภัย') | (pd.isnull(df_vew['DANGER_TYPE_NAME']))]
        
    # Convert string into datetime
    df_vew['final_plant_date'] = pd.to_datetime(df_vew['final_plant_date'])
    df_vew['START_DATE'] = pd.to_datetime(df_vew['START_DATE'],  errors='coerce')
    
    # Drop case [flood and loss_ratio==0] :== select only [not flood or loss_ratio!=0]
    df_vew = df_vew[(df_vew['loss_ratio']!=0) | pd.isna(df_vew['START_DATE'])]
     
    # Drop out of range loss ratio
#    df_vew = df_vew[(df_vew['loss_ratio']==0) | ((df_vew['loss_ratio']>=0.8) & (df_vew['loss_ratio']<=1))]
    df_vew = df_vew[(df_vew['loss_ratio']>=0) & (df_vew['loss_ratio']<=1)]

    df_vew.loc[pd.isna(df_vew['START_DATE']), ['START_DATE']] = df_vew[pd.isna(df_vew['START_DATE'])]['final_plant_date'] + datetime.timedelta(90)
    
    raster = rasterio.open(os.path.join(root_raster_sentinel1, f"LSVVS{raster_strip_id}_2017-2019"))
    list_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
    
    
    df_vew = df_vew[df_vew['START_DATE']>(list_date[0]+datetime.timedelta(12*flood_map_window_size))]
    df_vew = df_vew[df_vew['START_DATE']<=(list_date[-1]-datetime.timedelta(12*flood_map_window_size))]

    # นาปี
    df_vew = df_vew[((df_vew['final_plant_date'].dt.month>=5) & (df_vew['final_plant_date'].dt.month<=10))]
    
    df_vew = df_vew.reset_index(drop=True)    
    
    return df_mapping, df_vew

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

def save_parameters(threshold_data, filename_base, root_result, raster_strip_id):
    '''
    This function save threshold parameters and details in .json file and return of saved path (for num_log)
    where
        threshold_data : dictionary of save data
        root_result : save root 
    '''
    path_threshold_data = os.path.join(root_result, f"{raster_strip_id}", f"{filename_base}_({threshold_data['water_percentile']:.2f},{threshold_data['change_to_water_percentile']:.2f},{threshold_data['change_from_water_percentile']:.2f}).json")
    if not os.path.exists(os.path.dirname(path_threshold_data)):
        os.mkdir(os.path.dirname(path_threshold_data))
    json.dump(threshold_data, open(path_threshold_data, 'w'))
    return path_threshold_data

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
    valid_mask = np.zeros((height, width), 'uint8')
    
    # Get valid_mask, and list of band_date
    if init is None:
        list_date = []
        for band in range(num_bands):
            raster_band = raster.GetRasterBand(selected_band[0]+band+1)
            band_name = raster_band.GetDescription()
            date = band_name[-8:]
            list_date.append(date)
            
            raster_im = raster_band.ReadAsArray()
            valid_mask += (raster_im>0)
        
        valid_mask = (valid_mask>(0.9*num_bands)).astype('uint8')
        
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
            
            # Find and remove flood's pixels that change from water to land or already land (from otsu)
            row_not_flood, col_not_flood = np.nonzero((changed_from_water_area)+(water_im==0)) # (change from water to land map + not water map))
            im_flood[row_not_flood, col_not_flood] = 0
            im_flood[row_nvalid, col_nvalid] = 0     
        
        # Save flood map and continue for next loop
        im_previous = raster_im.copy()
        
        map_flood[band] = (im_flood>0).astype('uint8')
#        map_flood[band] = modeFilter(im_flood, size=5)
    
    if init==None:
        list_date = [datetime.datetime(int(item[0:4]), int(item[4:6]), int(item[6:8])) for item in list_date]
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
        if observed_date<=date:
            break
    return idx

def get_predict_and_ref(df_mapping, df_vew, map_flood, list_date, init_flood, flood_map_window_size):
    '''
    This function return predicted and ref loss ratio of every data in activity dataframe
    where
        df_mapping : mapping dataframe
        df_vew : all activities dataframe
        path_raster_flood : path of flood map raster
    '''
    ref = []
    predicted = []
    
    for start_date, df_vew_startdate_grp in df_vew.groupby(['START_DATE']):
#        break
        satellite_first_index = find_date_index(list_date, start_date-datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1
        satellite_last_index  = find_date_index(list_date, start_date+datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1
    
        satellite_first_index = max(0, satellite_first_index)
        satellite_last_index = min(satellite_last_index+1, len(map_flood))       
        
        for i in range(len(df_vew_startdate_grp)):
            # Get new_polygon_id
            df_vew_startdate_grp_selected = df_vew_startdate_grp.iloc[i]
            
            # Get row, col of satellite img
            rows = []
            cols = []
            for item in df_mapping.at[df_vew_startdate_grp_selected['new_polygon_id'], 'row_col']:
                rows.append(int(item.split('_')[0]))
                cols.append(int(item.split('_')[1]))
            
            selected_pixels = map_flood[satellite_first_index:satellite_last_index, rows, cols]  
            predicted_loss = np.any(selected_pixels, axis=0).astype('uint8')

            predicted_loss_ratio = predicted_loss.sum()/len(predicted_loss)
            ref_loss_ratio = df_vew_startdate_grp_selected['loss_ratio']
         
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