import os
from datetime import datetime
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import itertools
import random
import csv
from io import StringIO
import shutil
import json
import dask.dataframe as ddf
from numba import jit

import geopandas as gpd
from icecream import ic
import gdal
from gdalconst import *
import rasterio
from rasterio.transform import from_origin, guard_transform
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio import plot
from rasterio import Affine

import matplotlib.pyplot as plt
import seaborn as sns

from zkyhaxpy import io_tools, console_tools, json_tools, gis_tools, pd_tools, np_tools, dttm_tools, dict_tools
# from zkyhaxpy import colab_tools
from croppy import plant_info_tools, pixval_tools
#%%
def get_test_dataset_f(in_arr_last_ext_act_id, list_last_ext_act_id_for_test=[8, 9]):
    out_arr = np.where(
        np.isin(in_arr_last_ext_act_id, list_last_ext_act_id_for_test),
        1,
        0
    )
    return out_arr

def get_list_files(root_hls, scene_id):
    list_files = []
    for r, _, files in os.walk(os.path.join(root_hls, scene_id)):
        for file in files:
            if file.endswith(".tif"):
                list_files.append(os.path.join(r, file))
    df_list_files = io_tools.filepaths_to_df(list_files)
    return df_list_files

def get_year_list_band_id(img_date_from, img_date_to, dict_img_date):
    '''
    Get a list of tuple (year, list of band ids) according to img_date_from and img_date_to
    '''
    list_out = []
    list_img_date_from = get_raster_band_id_of_img_date(img_date_from, dict_img_date)
    list_img_date_to = get_raster_band_id_of_img_date(img_date_to, dict_img_date)


    if len(list_img_date_from) == 1:
        year_from = list_img_date_from[0]['year']
        band_id_from = list_img_date_from[0]['raster_band_id']
    elif len(list_img_date_from) > 1:
        year_from = list_img_date_from[0]['year']
        band_id_from = list_img_date_from[0]['raster_band_id']

    if len(list_img_date_to) == 1:
        year_to = list_img_date_to[0]['year']
        band_id_to = list_img_date_to[0]['raster_band_id']
    elif len(list_img_date_to) > 1:
        year_to = list_img_date_to[-1]['year']
        band_id_to = list_img_date_to[-1]['raster_band_id']

    if year_from == year_to:
        #Need only 1 year of raster
        year = year_from
        list_raster_band_id = list(range(band_id_from, band_id_to + 1))
        list_raster_band_desc_idx = list(range(band_id_from - 1, band_id_to))
        list_raster_band_desc = dict_img_date[year][list_raster_band_desc_idx]
        list_out.append({'year':year, 'list_raster_band_id':list_raster_band_id, 'list_raster_band_desc':list_raster_band_desc})
    else:
        #First year
        year = year_from
        list_raster_band_id = list(range(band_id_from, len(dict_img_date[year]) + 1))
        list_raster_band_desc_idx = list(range(band_id_from - 1, len(dict_img_date[year])))
        list_raster_band_desc = dict_img_date[year][list_raster_band_desc_idx]
        list_out.append({'year':year, 'list_raster_band_id':list_raster_band_id, 'list_raster_band_desc':list_raster_band_desc})

        #Last year
        year = year_to
        list_raster_band_id = list(range(1, band_id_to + 1))
        list_raster_band_desc_idx = list(range(0, band_id_to))
        list_raster_band_desc = dict_img_date[year][list_raster_band_desc_idx]
        list_out.append({'year':year, 'list_raster_band_id':list_raster_band_id, 'list_raster_band_desc':list_raster_band_desc})

    return list_out

def get_raster_band_id_of_img_date(img_date, dict_img_date):
    '''
    Get band id for given img_date
    If given image date is found in more than 1 year, 
    '''

    list_out = []
    if type(img_date) != np.datetime64:
        img_date = np.datetime64(img_date)

    for year in dict_img_date.keys():
        list_band_id = np.argwhere(dict_img_date[year]==img_date)
        if len(list_band_id) > 0:            
            raster_band_id = list_band_id[0][0] + 1
            raster_band_desc_idx = list_band_id[0][0]        
            list_out.append({'year':year, 'raster_band_id':raster_band_id, 'raster_band_desc_idx':raster_band_desc_idx})
    
    assert(len(list_out) > 0)
    
    return list_out

def combine_arr_from_2_years(arr_first, arr_last, list_img_date_first, list_img_date_last, overlap=True, overlap_func=np.nanmax):
    '''
    Combine arrays from 2 years of raster values 
    '''
    len_first = len(arr_first)
    len_last = len(arr_last)
    
    if overlap==True:
        arr_overlap = np.nanmax([arr_first[[-1]], arr_last[[0]]], axis=0)
        arr_out = np.concatenate([arr_first[:-1], arr_overlap, arr_last[1:]])
        list_img_date_out = [*list_img_date_first[:-1], *list_img_date_last]
    else:
        arr_out = np.concatenate([arr_first, arr_last])
        list_img_date_out = [*list_img_date_first, *list_img_date_last]
    
    return (arr_out, list_img_date_out)

def get_list_task():
    dict_mapping = {}
    for file in os.listdir(r"F:\CROP-PIER\CROP-WORK\!hls-prep\df_ext_act_id_x_hls_row_col\log"):
        if not file.split(".")[0].split("_")[-2] in dict_mapping.keys():
            dict_mapping[file.split(".")[0].split("_")[-2]] = [file.split(".")[0].split("_")[-3][1:]]
        else:
            dict_mapping[file.split(".")[0].split("_")[-2]].append(file.split(".")[0].split("_")[-3][1:])
    for key in dict_mapping.keys():
        dict_mapping[key] = np.unique(dict_mapping[key]).tolist()
    available_hls = os.listdir(root_hls)
    list_task = []
    for pathrow in available_hls:
        [list_task.append(p) for p in dict_mapping[pathrow]]
    list_task = np.unique(list_task).tolist()
    return list_task
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_hls = r"F:\CROP-PIER\CROP-WORK\!hls-prep\ndvi-16d-noise-reduct-splitted"
root_hls_vrt = r"C:\Users\PongporC\Desktop\temp\HLS-NDVI"
root_mapping = r"F:\CROP-PIER\CROP-WORK\!hls-prep\df_ext_act_id_x_hls_row_col"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\hls_pixval_ndvi_16d_nr"

gdf_grid_tiles = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!hls-prep\hls_tiles\hls_tiles.shp")
gdf_grid_tiles['scene_id'] = 'T' + gdf_grid_tiles['Name']
gdf_grid_tiles.index = gdf_grid_tiles['scene_key']
dict_scene_key_x_scene_id = gdf_grid_tiles['scene_id'].to_dict()

LIST_REQUIRED_PLANT_INFO_COL = [
    'ext_profile_center_id', 'plant_province_code', 'plant_amphur_code',
    'plant_tambon_code', 'ext_act_id', 'breed_code',       
    'total_actual_plant_area_in_wa', 'final_plant_year',
    'final_plant_month', 'final_plant_date',
    'in_season_rice_f', 
    'loss_type_class', 'loss_ratio',
    'loss_ratio_class', 'target_f_drought', 'target_f_flood',
    'target_f_other', 'tambon_pcode', 'test_dataset_f'
]
NBR_IMAGES_BEFORE_PLANT_DATE = 2
NBR_IMAGES = 14
list_ndvi_col_nm = [f't{np.char.zfill(str(i), 2)}' for i in range(1, NBR_IMAGES + 1)]
list_p = np.unique([file.split(".")[0][-7:-5] for file in os.listdir(root_vew)]).tolist()
#%% Get list_task (from available pathrow)
list_task = get_list_task()
#%%
for prov_cd in tqdm(list_task):
    print(prov_cd)
    path_save = os.path.join(root_save, f"df_pixval_hls_ndvi_16d_nr_p{prov_cd}.parquet")
    if os.path.exists(path_save):
        continue
    
    #Get list of files
    list_files_vew = [os.path.join(root_vew, file) for file in os.listdir(root_vew) if (file.split(".")[0][-7:-5] == prov_cd) and (file.split(".")[0][-4:] in ["2018", "2019", "2020"])]
    list_files_df_row_col = io_tools.get_list_files_re(os.path.join(root_mapping, f'p{prov_cd}'), '.*[2018|2019|2020].parquet')
    if len(list_files_df_row_col) == 0:
        print(f'There is no pixel available in f{prov_cd}. Skip this province.')
        continue
    
    #Load data into dataframes
    df_row_col = ddf.read_parquet( list_files_df_row_col)
    df_row_col = df_row_col.compute()
    df_vew = pd_tools.read_parquets(list_files_vew)
    df_vew.columns = df_vew.columns.str.lower()

    df_vew['ext_act_id'] = df_vew['ext_act_id'].astype(np.int64)
    df_vew['ext_profile_center_id'] = df_vew['ext_profile_center_id'].astype(np.int64)
    df_vew = plant_info_tools.get_last_digit_ext_act_id(df_vew).copy()
    df_vew['test_dataset_f'] = get_test_dataset_f(df_vew['last_digit_ext_act_id'].values)

    if str(df_vew['final_plant_date'].dtype) != 'datetime64[ns]':
        print('Convert final_plant_date into np.datetime64')
        df_vew['final_plant_date'] = df_vew['final_plant_date'].astype(np.datetime64)
    
    df_vew['danger_date'] = df_vew['danger_date'].astype(np.datetime64)
    df_vew['total_danger_area_in_wa'] = df_vew['total_danger_area_in_wa'].astype(np.float32)
    df_vew = plant_info_tools.get_loss_info(df_vew, danger_date_col='danger_date')
    df_vew[['plant_province_code', 'plant_amphur_code', 'plant_tambon_code']] = df_vew[['plant_province_code', 'plant_amphur_code', 'plant_tambon_code']].astype(int)
    df_vew = plant_info_tools.get_tambon_pcode(df_vew)
    nbr_ext_act_id_all = len(df_vew['ext_act_id'].unique())

    #Fill missing value
    df_vew['breed_code'] = df_vew['breed_code'].fillna(-99999)

    #Keep only required columns
    df_vew = df_vew.loc[:, LIST_REQUIRED_PLANT_INFO_COL].copy()

    #Filter only in-season-rice
    df_vew = plant_info_tools.filter_only_in_season_rice(df_vew).copy()
    df_vew = df_vew[df_vew['final_plant_date'] >= '2018-04-01'].copy()
    nbr_ext_act_id_in_season_rice = len(df_vew['ext_act_id'].unique())

    #Join plant info with pixel row col
    df_vew_joined = df_vew.merge(df_row_col, how='left', left_on='ext_act_id', right_on='ext_act_id').copy()
    df_vew_joined = df_vew_joined.dropna(subset=['scene_key']).copy()
    df_vew_joined[['row', 'col']] = df_vew_joined[['row', 'col']].fillna(-999).astype(np.int16)
    nbr_ext_act_id_in_season_rice_w_pixel = len(df_vew_joined['ext_act_id'].unique())
    nbr_pixel_all = len(df_vew_joined)

    #Get NDVI pixel values
    list_df_hls_v14_pixlvl = []
    for scene_key, df_vew_joined_curr_scene in tqdm(df_vew_joined.groupby('scene_key'), 'Iterate by scene'):
        scene_id = dict_scene_key_x_scene_id[scene_key]
        print('###########################################################')
        print(f'Processing for scene_id: {scene_id}...')

        #Download NDVI
        df_vew_joined_curr_scene['rep_plant_date_16d'] = dttm_tools.date_to_date_ndays(df_vew_joined_curr_scene['final_plant_date'])   
        
        # Get dict_img_date for current scene_id
        df_ndvi_16d_files = get_list_files(root_hls, scene_id)
        if len(df_ndvi_16d_files) == 0:
            print(f'scene_id: {scene_id} has no image prepared. Skip this scene.')
            continue

        df_ndvi_16d_files['img_date'] = df_ndvi_16d_files['file_nm'].str.split('.', expand=True)[0].str.slice(-10).str.replace("-", "")
        df_ndvi_16d_files['img_date'] = dttm_tools.yyyymmdd_to_numpy_datetime(df_ndvi_16d_files['img_date'].values)
        df_ndvi_16d_files['year'] = df_ndvi_16d_files['folder_nm']
        dict_img_date = {}
        for year, df_ndvi_16d_files_curr in df_ndvi_16d_files.groupby('year'):
            dict_img_date[year] = df_ndvi_16d_files_curr['img_date'].values.astype(dtype='datetime64[D]')

        #Iterate by rep_plant_date_16d
        for rep_plant_date_16d, df_vew_joined_curr_scene_rep_plant_date_16d in tqdm(df_vew_joined_curr_scene.groupby('rep_plant_date_16d'), 'Iterate by rep_plant_date_16d'):
            print(f'Processing for rep_plant_date_16d: {rep_plant_date_16d}...')
            
            #Get img_date_from & img_date_to for rep_plant_date_16d
            img_date_from = rep_plant_date_16d - np.timedelta64(NBR_IMAGES_BEFORE_PLANT_DATE * 16, 'D')
            img_date_to = rep_plant_date_16d + np.timedelta64((NBR_IMAGES - NBR_IMAGES_BEFORE_PLANT_DATE - 1) * 16, 'D')
            #Get list of year x band id for img_date_from & img_date_to
            try:
                list_year_band_id = get_year_list_band_id(img_date_from, img_date_to, dict_img_date)
            except:
                print(f'Skip {rep_plant_date_16d}...')
                continue
            
            #Get arr_ndvi according to rep_plant_date_16d
            if len(list_year_band_id) == 1:
                #If arr_ndvi needs to get from 1 year, simply read it from 1 raster
                dict_year_band_id = list_year_band_id[0]
                img_year = dict_year_band_id['year']
                list_raster_band_id = dict_year_band_id['list_raster_band_id']
                list_raster_band_desc = dict_year_band_id['list_raster_band_desc']
                with rasterio.open(os.path.join(root_hls_vrt, scene_id, f"{scene_id}_{img_year}.vrt"), mode='r') as ds_ndvi:
                    arr_ndvi = ds_ndvi.read(list_raster_band_id)

            elif len(list_year_band_id) == 2:
                #If arr_ndvi needs to get from 2 years, read 2 rasters seperately then combine it together
                #First year
                dict_year_band_id_first = list_year_band_id[0]
                img_year_first = dict_year_band_id_first['year']
                list_raster_band_id_first = dict_year_band_id_first['list_raster_band_id']
                list_raster_band_desc_first = dict_year_band_id_first['list_raster_band_desc']
                with rasterio.open(os.path.join(root_hls_vrt, scene_id, f"{scene_id}_{img_year_first}.vrt"), mode='r') as ds_ndvi:
                    arr_ndvi_first = ds_ndvi.read(list_raster_band_id_first)
                
                #Last year
                dict_year_band_id_last = list_year_band_id[1]
                img_year_last = dict_year_band_id_last['year']
                list_raster_band_id_last = dict_year_band_id_last['list_raster_band_id']
                list_raster_band_desc_last = dict_year_band_id_last['list_raster_band_desc']
                with rasterio.open(os.path.join(root_hls_vrt, scene_id, f"{scene_id}_{img_year_last}.vrt"), mode='r') as ds_ndvi:
                    arr_ndvi_last = ds_ndvi.read(list_raster_band_id_last)
                
                #Check if there is overlapping img_date between 2 years (possible in case 16 days ndvi are calculated in from different years)
                overlap = len(set(list_raster_band_desc_first).intersection(set(list_raster_band_desc_last))) > 0

                #Combine 2 years of arr_ndvi
                arr_ndvi, list_raster_band_desc = combine_arr_from_2_years(arr_ndvi_first, arr_ndvi_last, list_raster_band_desc_first, list_raster_band_desc_last, overlap=overlap)
                
            #Check if index still unique
            assert(df_vew_joined_curr_scene_rep_plant_date_16d.index.duplicated().sum() == 0)

            #Extract pixel values from arr_ndvi and append them to original dataframe
            arr_row_col = df_vew_joined_curr_scene_rep_plant_date_16d[['row', 'col']].values
            arr_pixval = pixval_tools.extract_values_from_3d_array_with_row_col_numba(arr_ndvi, arr_row_col)   
            # arr_ndvi[:, arr_row, arr_col].T is 2X faster and easier to use (3.66 µs, 7.75 µs)
            df_hls_v14_pixlvl_curr = pd.concat([df_vew_joined_curr_scene_rep_plant_date_16d, pd.DataFrame(arr_pixval[:, :14], index=df_vew_joined_curr_scene_rep_plant_date_16d.index, columns=list_ndvi_col_nm)], axis=1).copy()
            list_df_hls_v14_pixlvl.append(df_hls_v14_pixlvl_curr)

    #Concat all scenes together
    df_hls_v14_pixlvl = pd.concat(list_df_hls_v14_pixlvl)
    del(list_df_hls_v14_pixlvl)

    # Save
    df_hls_v14_pixlvl.to_parquet(path_save)
