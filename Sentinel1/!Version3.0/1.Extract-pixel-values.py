import os
import re
import datetime
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
import pyproj
from shapely import wkt
from shapely.ops import transform
from rasterio.mask import mask
from tqdm import tqdm
from zkyhaxpy.dttm_tools import yyyymmdd_add_days
from icedumpy.df_tools import load_vew, clean_and_process_vew

warnings.filterwarnings("ignore")
#%% Pre-define global variables
SAT_TYPE = "S1AB"
PERIOD = 4
REPEAT_CYCLE = 6

ROOT_RASTER = r"G:\!PIER\!FROM_2TB\Complete_VV_separate"
ROOT_DF_VEW = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
ROOT_DF_MAPPING = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v3(at-true)"
ROOT_RASTER_MAPPING = r"G:\!PIER\!FROM_2TB\s1_pixel_rowcol_map"
ROOT_TEMP = rf"C:\Users\PongporC\Desktop\temp\{SAT_TYPE}"
ROOT_DF_FLOOD = rf"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\{SAT_TYPE.lower()}_flood_pixel_new"
ROOT_DF_NONFLOOD = rf"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\{SAT_TYPE.lower()}_nonflood_pixel_new"

os.makedirs(ROOT_TEMP, exist_ok=True)
os.makedirs(ROOT_DF_FLOOD, exist_ok=True)
os.makedirs(ROOT_DF_NONFLOOD, exist_ok=True)
LIST_IGNORED = [".xml", ".ini"]

PROJECT_47 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32647',  # destination coordinate system
    always_xy=True # Must have
).transform

PROJECT_48 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32648',  # destination coordinate system
    always_xy=True # Must have
).transform 
#%% Define functions (create_temp_raster)
raster_datetime = lambda file: datetime.datetime.strptime(os.path.splitext(file)[0].split("_")[1], "%Y-%m-%d")
path_row = lambda file: os.path.join(ROOT_RASTER_MAPPING, f"s1_pixel_row_map_{os.path.basename(os.path.dirname(file))}.tiff")
path_col = lambda file: os.path.join(ROOT_RASTER_MAPPING, f"s1_pixel_col_map_{os.path.basename(os.path.dirname(file))}.tiff")

def get_raster_files_info():
    list_path = []
    list_date = []
    list_strip_id = []
    list_sat_type = []
    for root, _, files in os.walk(ROOT_RASTER):
        for idx, file in enumerate(files):
            if file.endswith(tuple(LIST_IGNORED)):
                continue
            path_file = os.path.join(root, file)
            strip_id = os.path.basename(root)
            sat_type = os.path.basename(os.path.dirname(root))
            
            date = raster_datetime(file)
            list_path.append(path_file)
            list_date.append(date)
            list_strip_id.append(strip_id)
            list_sat_type.append(sat_type)
            
    df = pd.DataFrame(
            {"path_file":list_path,
             "date":list_date,
             "strip_id":list_strip_id,
             "sat_type":list_sat_type}, columns=["path_file", "date", "strip_id", "sat_type"]
    )
    
    df = df.sort_values(["date"])
    return df

def filter_raster_info(df, strip_id, sat_type):
    df = df.copy()
    df = df[df["strip_id"] == strip_id]
    if sat_type != "S1AB":
        df = df[df["sat_type"] == sat_type]
    return df

def create_vrt(path_vrt, list_path_raster, list_name=None):
    list_path_raster = [path_row(list_path_raster[0]), path_col(list_path_raster[0])]  + list_path_raster
    outds = gdal.BuildVRT(path_vrt, list_path_raster, separate=True)
    if list_name is not None:
        list_name = ["row", "col"] + list_name
        for channel, name in zip(range(1, outds.RasterCount+1), list_name):
            outds.GetRasterBand(channel).SetDescription(name)
    outds.FlushCache()
    del outds

def create_trs(path_dst, path_src, file_format='ENVI'):
    outds = gdal.Translate(path_dst, path_src, format=file_format)
    outds.FlushCache()
    del outds
    
def main_create_temp_raster(strip_id, sat_type):
    path_vrt = os.path.join(ROOT_TEMP, f"{sat_type}_{strip_id}.vrt")
    path_trs = os.path.join(ROOT_TEMP, f"{sat_type}_{strip_id}")

    df_info = get_raster_files_info()
    df_info_filtered = filter_raster_info(df_info, strip_id=strip_id, sat_type=sat_type)
    if sat_type == "S1AB":
        df_info_filtered = df_info_filtered[df_info_filtered["date"] >= datetime.datetime(2018, 6, 1)]
    
    create_vrt(path_vrt, df_info_filtered["path_file"].tolist(), (df_info_filtered["date"].dt.date.astype(str).str.split("-").str.join("") + "_" + df_info_filtered["sat_type"]).tolist())
    if not os.path.exists(path_trs):
        create_trs(path_trs, path_vrt, file_format='ENVI')
    return path_trs
#%% Define functions (prepare_vew_dataframe)
def get_anchor_date_for_nonflood(list_plant_date, raster_last_date, period, repeat_cycle):
    gap = timedelta(repeat_cycle*(period-1))
    anchor_date = []
    for plant_date in list_plant_date:
        try:
            random_date = timedelta(np.random.randint(0, min(180, (raster_last_date-gap-plant_date).days+1)))
        except:
            random_date = timedelta(min(180, (raster_last_date-gap-plant_date).days+1))
        anchor_date.append(plant_date + random_date)
    return anchor_date

def load_vew_flood_nonflood(p, raster_first_date, raster_last_date, period, repeat_cycle):
    gap = timedelta(repeat_cycle*(period-1))
    if type(p) is not list:
        p = [p]
    
    # Load and clean
    df_vew = load_vew(ROOT_DF_VEW, p)
    df_vew = clean_and_process_vew(df_vew)
    
    # Calculate polygon area
    df_vew["polygon_area"] = df_vew["final_polygon"].apply(lambda val: create_polygon_from_wkt(val, to_crs="epsg:32647").area)
    
    # Separate flood and nonflood
    df_vew_flood = df_vew.loc[df_vew["DANGER_TYPE_NAME"]=="อุทกภัย"]
    df_vew_nonflood = df_vew.loc[~(df_vew["DANGER_TYPE_NAME"]=="อุทกภัย")]
    
    # Filter out out-of-range date
    df_vew_flood = df_vew_flood[df_vew_flood["START_DATE"] > (raster_first_date+gap)]
    df_vew_nonflood = df_vew_nonflood[df_vew_nonflood["final_plant_date"]+timedelta(30) > (raster_first_date+gap)]
    
    
    # Define anchor_date (Flood:START_DATE)
    df_vew_flood = df_vew_flood.assign(ANCHOR_DATE = df_vew_flood["START_DATE"])
    
    # Define anchor_date (Non:Flood)
    # Equation: 
    df_vew_nonflood = df_vew_nonflood.assign(ANCHOR_DATE = get_anchor_date_for_nonflood(df_vew_nonflood["final_plant_date"].tolist(),
                                                                                        raster_last_date,
                                                                                        period,
                                                                                        repeat_cycle))
    
    return df_vew_flood, df_vew_nonflood

def get_list_p_from_strip_id(strip_id):
    list_p = list()
    for file in os.listdir(ROOT_DF_MAPPING):
        file = file.split(".")[0].split("_")
        if file[-1][1:] == strip_id:
            list_p.append(file[-2][1:])
    return list_p

def main_prepare_vew_dataframe(strip_id, p, path_trs, period, repeat_cycle):
    raster = rasterio.open(path_trs)
    raster_first_date = datetime.datetime.strptime(raster.descriptions[2][:8], "%Y%m%d")
    raster_last_date = datetime.datetime.strptime(raster.descriptions[-1][:8], "%Y%m%d")
    df_vew_flood, df_vew_nonflood = load_vew_flood_nonflood(p, raster_first_date, raster_last_date, period, repeat_cycle)
    return df_vew_flood, df_vew_nonflood
#%% Define functions (extract_pixel_values)
def create_polygon_from_wkt(wkt_polygon, crs="epsg:4326", to_crs=None):
    """
    Create shapely polygon from string (wkt format) "MULTIPOLYGON(((...)))"
    https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects/127432#127432

    Parameters
    ----------
    wkt_polygon: str
        String of polygon (wkt format).
    crs: str
        wkt_polygon's crs (should be "epsg:4326").
    to_crs: str (optional), default None
        Re-project crs to "to_crs".

    Examples
    --------
    >>> create_polygon_from_wkt(wkt_polygon)
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32647")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32648")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32660")
    >>> create_polygon_from_wkt(wkt_polygon, crs="epsg:32647", to_crs="epsg:4326")

    Returns
    -------
    Polygon
    """
    polygon = wkt.loads(wkt_polygon)
    if to_crs is not None:
        if crs == "epsg:4326":
            if to_crs == "epsg:32647":
                polygon = transform(PROJECT_47, polygon)
            elif to_crs == "epsg:32648":
                polygon = transform(PROJECT_48, polygon)
        else:
            project = pyproj.Transformer.from_crs(
                crs,     # source coordinate system
                to_crs,  # destination coordinate system
                always_xy=True # Must have
            ).transform
            polygon = transform(project, polygon)
    return polygon

def get_arr_ideal_band_desc(raster, repeat_cycle, backward_cycles=20, forward_cycles=20):
    '''
    Get all ideal band desc of given raster.     
    
    Parameters
    ----------------------
    raster : rasterio raster
        A raster with band descriptions. First and Second band desc must be row and column accordingly.
    Band desc for image must be string date in YYYYMMDD format.
    
    repeat_cycle : int
        No. of days of repeating cycle.
        
    backward_cycles : int (optional), default 20
        No. of cycles to get ideal band desc backward before the first image date in raster.
    
    forward_cycles : int (optional), default 20
        No. of cycles to get ideal band desc forward after the last image date in raster.
        
    Return
    All ideal bands desc of given raster
    ----------------------
    
    '''
    arr_all_actual_band_desc = np.array([item[:8] for item in raster.descriptions])
    assert(arr_all_actual_band_desc[0]=='row')
    assert(arr_all_actual_band_desc[1]=='col')
    
    list_ideal_band_desc = ['row', 'col']
    img_first_date = str(np.min(arr_all_actual_band_desc[2:].astype(int)))
    img_last_date = str(np.max(arr_all_actual_band_desc[2:].astype(int)))
    
    ideal_band_desc = yyyymmdd_add_days(img_first_date, -repeat_cycle * backward_cycles)
    
    while ideal_band_desc <= yyyymmdd_add_days(img_last_date, repeat_cycle * forward_cycles):
        list_ideal_band_desc.append(ideal_band_desc)
        ideal_band_desc = yyyymmdd_add_days(ideal_band_desc, repeat_cycle)

    return np.array(list_ideal_band_desc)

def get_target_band_structure(arr_all_ideal_band_desc, arr_all_actual_band_desc, X_before_date, Y_after_date):
    '''
    Get target band structure for extracting values.
    Target band structure consists of 
    1) Target ideal band descriptions
    2) Target actual band descriptions
    3) Target actual band id
    
    Parameters
    ------------------
    arr_ideal_band_desc : Numpy.array(str)
        An arry of ideal band descriptions
    
    X_before_date : str
        A string of date in yyyymmdd to be used as starting date to extract pixel values
        
    Y_after_date : str
        A string of date in yyyymmdd to be used as ending date to extract pixel values
    
    Return
    ------------------
    A tuple (arr_target_ideal_band_desc, arr_target_actual_band_desc, arr_target_actual_band_id)
    arr_target_ideal_band_desc : numpy array (str)
        An array of target ideal band descriptions that match given X_before_date and Y_after_date
    arr_target_actual_band_desc : numpy array (str)
        An array of target actual band descriptions that match given X_before_date and Y_after_date
    arr_target_actual_band_id : numpy array (int)
        An array of target ideal band id that match given X_before_date and Y_after_date            
    '''
    
    arr_target_ideal_band_desc = np.where((np.isin(arr_all_ideal_band_desc, ['row', 'col']))|((arr_all_ideal_band_desc >= X_before_date) & (arr_all_ideal_band_desc <= Y_after_date)), arr_all_ideal_band_desc, np.nan)
    arr_target_ideal_band_desc = arr_target_ideal_band_desc[arr_target_ideal_band_desc != 'nan']
    arr_target_actual_band_desc = arr_target_ideal_band_desc[np.isin(arr_target_ideal_band_desc, arr_all_actual_band_desc)]
    
    arr_target_actual_band_id = np.where((arr_all_actual_band_desc == 'row')  | (arr_all_actual_band_desc == 'col')  | (np.isin(arr_all_actual_band_desc, arr_target_actual_band_desc)), range(1, len(arr_all_actual_band_desc) + 1), np.nan)
    arr_target_actual_band_id = arr_target_actual_band_id[~np.isnan(arr_target_actual_band_id)].astype(int)
    return arr_target_ideal_band_desc, arr_target_actual_band_desc, arr_target_actual_band_id

def extract_pixel_values(anchor_date,
                         X_before,
                         Y_after,
                         wkt_polygon,
                         path_raster,
                         repeat_cycle,
                         arr_ideal_band_desc,                         
                         unit_of_XY="day",
                         column_nm_format='t**',
                         other_info=None,
                         nodata_out_of_polygon=-999,
                         **masking_kwargs):
    
    raster = rasterio.open(path_raster)
    
    #get masking params
    all_touched = masking_kwargs.get("all_touched", True)
    crop = masking_kwargs.get("crop", True)

    arr_all_actual_band_desc = np.array([item[:8] for item in raster.descriptions])
    
    #validate unit of XY
    assert(unit_of_XY in ['day', 'period'])    

    #calculate X_before_days & Y_after_days
    if unit_of_XY=='period':
        X_before_days = X_before * repeat_cycle
        Y_after_days = Y_after * repeat_cycle
    elif unit_of_XY=='day':
        X_before_days = X_before
        Y_after_days = Y_after

    #Get starting date & ending date of image to be extracted. First day of Y is the anchor date.    
    X_before_date = yyyymmdd_add_days(anchor_date, -X_before_days)    
    Y_after_date = yyyymmdd_add_days(anchor_date, Y_after_days - 1)

    #get band structure
    arr_target_ideal_band_desc, arr_target_actual_band_desc, arr_target_actual_band_id = get_target_band_structure(arr_ideal_band_desc, arr_all_actual_band_desc, X_before_date, Y_after_date)
    
    if len(arr_target_actual_band_id) > 2:
        assert('row' in arr_target_actual_band_desc)
        assert('col' in arr_target_actual_band_desc)
        try:
            #create shapely polygon and convert to same crs as raster       
            to_crs = raster.crs["init"]
            polygon = create_polygon_from_wkt(wkt_polygon, to_crs=to_crs)

            #extract pixel values of matched date.
            arr_pixel_values, _ = mask(raster, polygon, crop=crop, all_touched=all_touched, indexes=list(arr_target_actual_band_id), nodata=nodata_out_of_polygon)
            arr_pixel_values = arr_pixel_values[arr_pixel_values != nodata_out_of_polygon]

            df_pixval = pd.DataFrame(arr_pixel_values.reshape(len(arr_target_actual_band_desc), -1).T, columns=arr_target_actual_band_desc)
            
            # Add other
            if other_info is not None:
                df_pixval = df_pixval.assign(**{index:other_info[index] for index in other_info.index})
                df_pixval = df_pixval.reindex(columns=other_info.index.to_list() + list(arr_target_ideal_band_desc))
            else:
                df_pixval = df_pixval.reindex(columns=list(arr_target_ideal_band_desc))
            df_pixval['row'] = df_pixval['row'].astype(int)
            df_pixval['col'] = df_pixval['col'].astype(int)
                
            #reformat column names 
            if re.match('^t(\*+)$', column_nm_format):
                n_zfill = sum([1 for char in column_nm_format if char == '*'])  
                if other_info is not None:
                    df_pixval.columns = other_info.index.to_list() + ['row', 'col'] + [''.join(['t', str(n).zfill(n_zfill)]) for n in range(1, len(arr_target_ideal_band_desc) -1)]
                else:
                    df_pixval.columns = ['row', 'col'] + [''.join(['t', str(n).zfill(n_zfill)]) for n in range(1, len(arr_target_ideal_band_desc) -1)]
            return ('Success', df_pixval)
        except Exception as e:
            return (str(e), pd.DataFrame())
    else:
        return ('No image', pd.DataFrame())

def main_extract_pixel_values(df, path_raster, arr_ideal_band_desc, other_info_columns=None):
    list_message = []
    list_df_pixel_values = []
    for index in df.index:
        anchor_date = "".join(str(df.at[index, "ANCHOR_DATE"].date()).split("-"))
        wkt_polygon = df.at[index, "final_polygon"]
        if other_info_columns is not None:
            other_info = df.loc[index, other_info_columns]
        
        message, df_pixel_values = extract_pixel_values(
                 anchor_date=anchor_date,
                 X_before=PERIOD,
                 Y_after=PERIOD,
                 wkt_polygon=wkt_polygon,
                 path_raster=path_raster,
                 repeat_cycle=REPEAT_CYCLE,
                 arr_ideal_band_desc=arr_ideal_band_desc,                         
                 unit_of_XY="period",
                 column_nm_format='t*',
                 other_info=other_info,
                 nodata_out_of_polygon=-999,
                 all_touched=True,
                 crop=True
             )
        
        list_message.append(message)
        list_df_pixel_values.append(df_pixel_values)
    return list_message, list_df_pixel_values
#%% Main
def check_all_files_exists(strip_id):
    list_p = get_list_p_from_strip_id(strip_id)
    if len([file for file in os.listdir(ROOT_DF_FLOOD) if file.split(".")[0].split("_")[1:]]) == len(list_p):
        if len([file for file in os.listdir(ROOT_DF_NONFLOOD) if file.split(".")[0].split("_")[1:]]) == len(list_p):
            return True
    return False
    
def main_each_strip_id(strip_id):
#    tqdm.write(f"Creating temp raster: {strip_id}")
    if not check_all_files_exists(strip_id):
        path_raster = main_create_temp_raster(strip_id, sat_type=SAT_TYPE)
        raster = rasterio.open(path_raster)
        arr_ideal_band_desc = get_arr_ideal_band_desc(raster=raster, repeat_cycle=REPEAT_CYCLE)
        
        list_p = get_list_p_from_strip_id(strip_id)
        pbar = tqdm(list_p)
        for p in pbar:
            pbar.set_description(f"Processing: p{p}")
            path_df_flood = os.path.join(ROOT_DF_FLOOD, f"df_{SAT_TYPE.lower()}_flood_pixel_p{p}_s{strip_id}.parquet")
            path_df_nonflood = os.path.join(ROOT_DF_NONFLOOD, f"df_{SAT_TYPE.lower()}_nonflood_pixel_p{p}_s{strip_id}.parquet")
            if os.path.exists(path_df_flood) and os.path.exists(path_df_nonflood):
                #tqdm.write(f"{path_df_flood} and {path_df_nonflood} already exist (skip)")
                continue
            
            df_vew_flood, df_vew_nonflood = main_prepare_vew_dataframe(strip_id, p, path_raster, PERIOD, REPEAT_CYCLE)
            
            # Flood
            if not os.path.exists(path_df_flood) and len(df_vew_flood) != 0:
                #tqdm.write(f"START: {os.path.basename(path_df_flood)}")
                list_message, list_df_pixel_values = main_extract_pixel_values(df_vew_flood, path_raster, arr_ideal_band_desc, 
                                                                               other_info_columns=['new_polygon_id', 'PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'ext_act_id', 'final_polygon', 'final_plant_date', 'ANCHOR_DATE', 'loss_ratio'])
                df = pd.concat(list_df_pixel_values)
                df.to_parquet(path_df_flood)
            else:
                df = pd.DataFrame()
                df.to_parquet(path_df_flood)
                
            # Non-flood
            if not os.path.exists(path_df_nonflood) and len(df_vew_nonflood) != 0:
                #tqdm.write(f"START: {os.path.basename(path_df_nonflood)}")
                list_message, list_df_pixel_values = main_extract_pixel_values(df_vew_nonflood, path_raster, arr_ideal_band_desc, 
                                                                               other_info_columns=['new_polygon_id', 'PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'ext_act_id', 'final_polygon', 'final_plant_date', 'ANCHOR_DATE', 'loss_ratio'])
                df = pd.concat(list_df_pixel_values)
                df.to_parquet(path_df_nonflood)
            else:
                df = pd.DataFrame()
                df.to_parquet(path_df_nonflood)
        del raster
        os.remove(path_raster)
#%%
if __name__=="__main__":
    # Everything here is global too
    # LIST_STRIP_ID = os.listdir(os.path.join(ROOT_RASTER, SAT_TYPE))
    # LIST_STRIP_ID = ["302", "303", "304", "305", "306", "401", "402", "403"]
    # LIST_STRIP_ID = ["101", "102", "103", "104", "105", "106", "107", "108", "109", 
                     # "201", "202", "203", "204", "205", "206", "207", "208",
                     # "301"]
    LIST_STRIP_ID = ["402"]
    PBAR = tqdm(LIST_STRIP_ID)
    for STRIP_ID in PBAR:
        PBAR.set_description(f"Processing: s{STRIP_ID}")
        main_each_strip_id(STRIP_ID)
#%%