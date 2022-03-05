import os
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from itertools import groupby
import matplotlib.pyplot as plt
#%%
# Find max consecutive 1 (https://codereview.stackexchange.com/questions/138550/count-consecutive-ones-in-a-binary-list)
def len_iter(items):
    return sum(1 for _ in items)

def consecutive_one(data):
    return max(len_iter(run) for val, run in groupby(data) if val)

@jit(nopython=True)
def interp_numba(arr_ndvi):
    '''
    Interpolate an array in both directions using numba.
    (From     P'Tee+)

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
        arr_ndvi_row_not_nan_idx = np.argwhere(
            ~np.isnan(arr_ndvi_row)).flatten()
        if len(arr_ndvi_row_not_nan_idx) > 0:
            arr_ndvi_row_not_nan_values = arr_ndvi_row[arr_ndvi_row_not_nan_idx]
            arr_ndvi[n_row] = np.interp(
                arr_ndvi_row_idx, arr_ndvi_row_not_nan_idx, arr_ndvi_row_not_nan_values)

    arr_ndvi = arr_ndvi.astype(np.float32)
    return arr_ndvi

def convert_power_to_db(df, columns):
    df = df.copy()

    # Replace negatives with nan
    for col in columns:
        df.loc[df[col] <= 0, col] = np.nan

    # Drop row with mostly nan
    df = df.loc[df[columns].isna().sum(axis=1) < 10]

    # Interpolate nan values
    df[columns] = interp_numba(df[columns].values)

    # Convert power to dB
    df[columns] = 10*np.log10(df[columns])
    return df

def extract_features(df, columns, label):
    # Loop for each group (group by ext_act_id)
    list_data = []
    columns_temp = columns.copy()
    
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        # Get basic info
        df_grp_info = df_grp.iloc[:, 15:-1]
        
        # If too early (before July), skip first growth stage (first 40 days)
        if (df_grp.iloc[0]["final_plant_date"].month < 7) and not (columns == ['t0', 't1', 't2']):
            columns = columns_temp[6:]
        
        # Find which "period" (1 or 2) gives min(diff+backscatter)
        periods = int(np.nanargmin([
            (df_grp[columns].diff(periods=1, axis=1)+df_grp[columns]).min(axis=1).min(),
            (df_grp[columns].diff(periods=2, axis=1)+df_grp[columns]).min(axis=1).min(),
        ])+1)
        
        # Find which column
        drop = df_grp[columns].diff(periods=periods, axis=1)
        coef = drop+df_grp[columns]
        flood_column = int(coef.min().idxmin()[1:])
        
        # Get data (flood_column-lag -> flood_column-lag+8)
        columns_for_feature = [f"t{i}" for i in range(max(0, flood_column-periods), min(int(columns[-1][1:])+1, flood_column-periods+4))]
        
        # =============================================================================
        # Create dict of extracted features (plot-level)
        # =============================================================================
        dict_extracted_features = {
            **df_grp_info.to_dict(orient="list"),
            f"total_period_{label}": len(columns_for_feature)
        }
        # Thresholding range(-12, -19, -1)
        # list_thresholds = list(range(-12, -19, -1))
        list_thresholds = [-15, -17, -19]
        # Extract features for each threshold
        for threshold in list_thresholds:
            arr = (df_grp[columns_for_feature] < threshold).values.astype(int)
            # Strict
            list_consecutive_strict = []
            for i in range(arr.shape[0]):
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_strict.append(0)
                else:
                    list_consecutive_strict.append(consecutive_one(arr[i]))
            # Relax
            list_consecutive_relax = []
            for i in range(arr.shape[0]):
                # Change from [1, 0 ,1] to [1, 1, 1] (window size = 3)
                for j in range(1, arr.shape[1]-1):
                    if (arr[i, j-1] == 1) and (arr[i, j+1] == 1):
                        arr[i, j] = 1
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_relax.append(0)
                else:
                    list_consecutive_relax.append(consecutive_one(arr[i]))
        
            # Period = 12 days
            dict_extracted_features = {
                **dict_extracted_features,
                f"simple_index_s1a_under({threshold})_{label}":list_consecutive_strict,
                f"simple_index_s1a_under({threshold})_relax_{label}":list_consecutive_relax,
            }
        # After finish extracting features (every threshold)
        list_data.append(pd.DataFrame(dict_extracted_features))
    df = pd.concat(list_data, ignore_index=True)
    return df
#%%
root_df_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\simple_index_s1a_pixel_level_features_nrt(at-False)"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1a_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
os.makedirs(root_df_save, exist_ok=True)

# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]
#%%
# Define columns for each growth stage
columns = [f"t{i}" for i in range(15)]
columns_stg1 = [f"t{i}" for i in range(0, 3)]   # 0-35
columns_stg2 = [f"t{i}" for i in range(3, 8)]   # 36-95
columns_stg3 = [f"t{i}" for i in range(8, 10)]  # 96-119
columns_stg4 = [f"t{i}" for i in range(10, 15)] # 120-179
#%%
# (vew == temporal) by the way.
for file in os.listdir(root_vew)[2::3]:
    strip_id = file.split(".")[0][-3:]
    p_code = file.split(".")[0].split("_")[-2][1:]
    path_file = os.path.join(root_vew, file)
    for year in [2017, 2018, 2019, 2020]:
        print(file, p_code, year)
        path_save = os.path.join(root_df_save, f"df_simple_index_s1a_pixel_level_features_p{p_code}_s{strip_id}_y{year}.parquet")
        if os.path.exists(path_save):
            continue
        # Load data
        try:
            df = pd.read_parquet(path_file)
            df = df[df.PLANT_PROVINCE_CODE == int(p_code)] # Filter by p_code
            df = df[df.final_plant_year == year] # Filter by year
            df = pd.merge(df, df_rice_code, on="BREED_CODE", how="left")
        
            # Some cleaning
            df = df[df["in_season_rice_f"] == 1]
            df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]
            df = convert_power_to_db(df, columns)
        except:
            continue
        
        if len(df) == 0:
            continue
        
        # Main process
        df_stg1 = extract_features(df, columns_stg1, label="nrt_stg1")
        df_stg2 = extract_features(df, columns_stg1+columns_stg2, label="nrt_stg2")
        df_stg3 = extract_features(df, columns_stg1+columns_stg2+columns_stg3, label="nrt_stg3")
        df_stg4 = extract_features(df, columns_stg1+columns_stg2+columns_stg3+columns_stg4, label="nrt_stg4")
        
        # Join stg[1, 2, 3, 4]
        df_stg = pd.concat([df_stg1, df_stg2.iloc[:, 36:], df_stg3.iloc[:, 36:], df_stg4.iloc[:, 36:]], axis=1)
    
        # Save file
        if len(df_stg) != 0:
            df_stg.to_parquet(path_save)
                
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            

