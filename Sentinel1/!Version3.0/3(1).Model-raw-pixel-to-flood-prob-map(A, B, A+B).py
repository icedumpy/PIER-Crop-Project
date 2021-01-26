import os
import datetime
import numpy as np
import pandas as pd
import rasterio
from numba import jit
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from icedumpy.io_tools import save_model, load_model, save_h5, load_h5
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe, load_mapping
from icedumpy.geo_tools import create_tiff, create_vrt
#%% Define functions
@jit(nopython=True) 
def interp_numba(arr_ndvi):
    '''
    Interpolate an array in both directions using numba.
    (From P'Tee+)
    
    Parameters
    ---------
    arr_ndvi : numpy.array(np.float32)
        An array of NDVI values to be interpolated.
        
    Returns
    ---------
    arr_ndvi : numpy.array(np.float32)
        An array of interpolated NDVI values
    '''
    
    for n_row in range(arr_ndvi.shape[0]):       
        arr_ndvi_row = arr_ndvi[n_row]
        arr_ndvi_row_idx = np.arange(0, arr_ndvi_row.shape[0], dtype=np.int64)
        arr_ndvi_row_not_nan_idx = np.empty(0, dtype=np.int64)
        arr_ndvi_row_not_nan_idx = np.argwhere(~np.isnan(arr_ndvi_row)).flatten()
        if len(arr_ndvi_row_not_nan_idx) > 0:
            arr_ndvi_row_not_nan_values = arr_ndvi_row[arr_ndvi_row_not_nan_idx]           
            arr_ndvi[n_row] = np.interp(arr_ndvi_row_idx, arr_ndvi_row_not_nan_idx, arr_ndvi_row_not_nan_values)        
        
    arr_ndvi = arr_ndvi.astype(np.float32)  
    return arr_ndvi

def get_df_raster_info(path_raster, repeat_cycle):
    raster = rasterio.open(path_raster)
    df_raster_info = pd.concat([pd.DataFrame([dict(enumerate(description.split("_")))]) for description in raster.descriptions if not description in ["row", "col"]], ignore_index=True).rename(columns={0:"raster_date", 1:"sat_type"})
    df_raster_info["raster_date"] = pd.to_datetime(df_raster_info["raster_date"])
    df_raster_info["band"] = df_raster_info.index+3 # 1 == row, 2 == col
    df_raster_info = pd.merge(
        left=pd.DataFrame(pd.date_range(df_raster_info.iat[0, 0], df_raster_info.iat[-1, 0], freq=f"{repeat_cycle}D"), columns=["ideal_date"]),
        right=df_raster_info,
        left_on="ideal_date",
        right_on="raster_date",
        how="outer"
    )
    return df_raster_info

def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]

    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df

def get_df_train_test(root_df_flood, root_df_nonflood, root_df_mapping, p, strip_id):
    # All_touched = False
    # Tier = [1]
    
    # Load flood and non-flood dataframe
    df_flood = load_s1_flood_nonflood_dataframe(root_df_flood, p, strip_id=strip_id)
    df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p, strip_id=strip_id)

    # Replace negative data with nan
    df_flood = replace_zero_with_nan(df_flood)
    df_nonflood = replace_zero_with_nan(df_nonflood)

    # Drop row with nan (any)
    columns_pixel_values = df_flood.columns[df_flood.columns.str.contains(r'^t\d+$')].tolist()
    df_flood = df_flood.dropna(subset=columns_pixel_values)
    df_nonflood = df_nonflood.dropna(subset=columns_pixel_values)
    
    # Concat flood and non-flood and drop All_touched == True (Use All_touched=False)
    df_sample = pd.concat([df_flood, df_nonflood])
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id, list_p=list(map(str, df_sample["PLANT_PROVINCE_CODE"].unique())))
    df_sample = pd.merge(df_sample, df_mapping, how='inner', on=["new_polygon_id", "row", "col"], left_index=True)

    # Create label
    df_sample = df_sample.assign(label=(df_sample["loss_ratio"] != 0).astype("uint8"))
    
    # Drop duplicates
    df_sample = df_sample.drop_duplicates(subset=columns_pixel_values)
    
    # Drop Tier 2 polygon
    df_sample = df_sample[df_sample["tier"].isin([1])]
    
    # Select on loss_ratio = 0, between(0.8, 1.0)
    df_sample = df_sample.loc[(df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"].between(0.8, 1.0))]
    
    # Downsampling non-flood (same ext_act_id as flood)
    df_sample = df_sample.loc[  
        (df_sample["ext_act_id"].isin(np.random.choice(df_sample.loc[df_sample["label"] == 0, "ext_act_id"].unique(), len(df_sample.loc[df_sample["label"] == 1, "ext_act_id"].unique()), replace=False)))
        | (df_sample["label"] == 1)
    ]
    
    # Shuffle data
    df_sample = df_sample.sample(frac=1.0)    
    
    # Create train-test samples
    df_train = df_sample.loc[(~(df_sample["ext_act_id"]%10).isin([8, 9])) & ((df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"] >= 0.8))]
    df_test = df_sample.loc[((df_sample["ext_act_id"]%10).isin([8, 9])) & ((df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"] >= 0.8))]
        
    return df_train, df_test

def create_model(path_model, root_df_flood, root_df_nonflood, root_df_mapping, p, strip_id):
    df_train, df_test = get_df_train_test(root_df_flood, root_df_nonflood, root_df_mapping, p=None, strip_id=strip_id)
    columns_pixel_values = df_train.columns[df_train.columns.str.contains(r'^t\d+$')].tolist()
    model = RandomForestClassifier(min_samples_leaf=5, max_depth=10, min_samples_split=10,
                                   verbose=0, class_weight="balanced",
                                   n_jobs=-1, random_state=42)
    model.fit(df_train[columns_pixel_values].values, df_train["label"].values)
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    save_model(path_model, model)
    
    y_predict_prob = model.predict_proba(df_train[columns_pixel_values].values)
    fpr, tpr, threshold = metrics.roc_curve(df_train["label"].values, y_predict_prob[:, 1])
    dict_roc_params = {"fpr":fpr,
                       "tpr":tpr,
                       "threshold":threshold}
    save_h5(os.path.join(os.path.dirname(path_model), f"{strip_id}_RF_raw_pixel_values_roc_params.h5"),
            dict_roc_params)
    
    return model, dict_roc_params

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%% Define constant parameters
sat_type = "S1AB" # "S1A", "S1B", "S1AB"
root_df_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v4(at-False)"
root_raster = r"C:\Users\PongporC\Desktop\temp"
root_valid_mask = r"G:\!PIER\!FROM_2TB\s1_valid_mask"
root_model = "F:\CROP-PIER\CROP-WORK\Model\sentinel1"

if sat_type == "S1AB":
    repeat_cycle = 6
else:
    repeat_cycle = 12
#%% Define parameters
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
# for strip_id in ["304", "402", "403"]:
    model_strip_id = strip_id
    
    root_df_flood = os.path.join(root_df_sentinel1, f"{sat_type.lower()}_flood_pixel")
    root_df_nonflood = os.path.join(root_df_sentinel1, f"{sat_type.lower()}_nonflood_pixel")
    path_raster = os.path.join(root_raster, sat_type, f"{sat_type}_{strip_id}.vrt")
    path_raster_valid_mask = os.path.join(root_valid_mask, f"s1_valid_mask_{strip_id}.tif")
    
    if model_strip_id == "all":
         root_raster_flood = os.path.join(r"G:\!PIER\!Flood-map\Sentinel-1", sat_type, "Prob", f"{strip_id}-all")
         path_model = os.path.join(root_model, sat_type, "all_RF_raw_pixel_values.joblib")
    else:
        root_raster_flood = os.path.join(r"G:\!PIER\!Flood-map\Sentinel-1", sat_type, "Prob", strip_id)
        path_model = os.path.join(root_model, sat_type, strip_id, f"{strip_id}_RF_raw_pixel_values.joblib")    
    os.makedirs(root_raster_flood, exist_ok=True)
    #%% Load model
    if os.path.exists(path_model):
        if model_strip_id == "all":
            path_roc_params = os.path.join(os.path.dirname(path_model), "all_RF_raw_pixel_values_roc_params.h5")
        else:
            path_roc_params = os.path.join(os.path.dirname(path_model), f"{strip_id}_RF_raw_pixel_values_roc_params.h5")
        model = load_model(path_model)
        dict_roc_params = load_h5(path_roc_params)
    else:
        model, dict_roc_params = create_model(path_model, root_df_flood, root_df_nonflood, root_df_mapping, p=None, strip_id=strip_id)
    #%% Get row, col of rice
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[df_mapping["tier"].isin([1])]
    df_mapping = df_mapping.drop_duplicates(subset=["row", "col"])
    row_rice = df_mapping["row"].values.astype("int16")
    col_rice = df_mapping["col"].values.astype("int16")
    del df_mapping
    #%% Get df_raster_info
    df_raster_info = get_df_raster_info(path_raster, repeat_cycle)
    # df_raster_info = df_raster_info.loc[df_raster_info["ideal_date"] >= datetime.datetime(2019, 4, 1)]
    # df_raster_info = df_raster_info.reset_index(drop=True)
    # df_raster_info = df_raster_info.loc[df_raster_info["ideal_date"].between(datetime.datetime(2018, 6, 1), 
                                                                              # datetime.datetime(2019, 12, 31))]
    raster = rasterio.open(path_raster)
    #%%
    dict_raster_im = dict()
    
    # Len of data (vector size)s
    length_vector = (48//repeat_cycle)
    for i in range(0, len(df_raster_info)+1-length_vector):
        
        # =============================================================================
        # Load data from raster    
        # =============================================================================
        # List of wanted bands
        list_band = [int(df_raster_info.at[j, "band"]) if not np.isnan(df_raster_info.at[j, "band"]) else df_raster_info.at[j, "band"] for j in range(i, i+length_vector)]
        list_date = [df_raster_info.at[j, "ideal_date"] for j in range(i, i+length_vector)]
        
        # Skip if already exist
        path_raster_flood = os.path.join(root_raster_flood, str(list_date[len(list_date)//2].date())+".tif")
        if os.path.exists(path_raster_flood):
            continue
    
        # Show details
        print(f"{strip_id}: {str(list_date[len(list_date)//2].date())}")
        
        # Drop not wanted bands
        [dict_raster_im.pop(key) for key in [key for key in dict_raster_im.keys() if not key in list_date]]
        
        for idx, (band, date) in enumerate(zip(list_band, list_date)):
            if band in dict_raster_im.keys():
                continue
            if np.isnan(band):
                dict_raster_im[date] = np.full([len(row_rice)], np.nan).reshape(-1, 1)
            else:
                dict_raster_im[date] = raster.read(band)[row_rice, col_rice].reshape(-1, 1)
        
        data = np.hstack(list(dict_raster_im.values()))

        # Keep only not edge
        pos_not_edge = (data != 0).all(axis=1)
        row_rice_not_edge = row_rice[pos_not_edge]
        col_rice_not_edge = col_rice[pos_not_edge]
        data = data[pos_not_edge]
        
        # Interpolate nan data
        if np.isnan(data).any():
            data = interp_numba(data)

        # Create flood map
        y_pred = model.predict_proba(data)[:, 1]
        img = np.zeros((raster.height, raster.width), dtype="float32")
        img[row_rice_not_edge, col_rice_not_edge] = y_pred
        create_tiff(path_save=path_raster_flood,
                    im=np.expand_dims(img, axis=0),
                    projection=raster.crs.to_wkt(),
                    geotransform=raster.get_transform(),
                    drivername="GTiff",
                    dtype="float32"
                    )
        
    create_vrt(path_save=os.path.join(root_raster_flood, "flood_map.vrt"),
               list_path_raster=[os.path.join(root_raster_flood, file) for file in os.listdir(root_raster_flood) if file.endswith("tif")],
               list_band_name=[file.split(".")[0] for file in os.listdir(root_raster_flood) if file.endswith("tif")],
               src_nodata=0,
               dst_nodata=0)
#%% Okay..










