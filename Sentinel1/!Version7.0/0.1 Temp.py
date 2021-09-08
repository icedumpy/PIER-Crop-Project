import os
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
@jit(nopython=True)
def interp_numba(arr_ndvi):
    '''
    Interpolate an array in both directions using numba.
    (From     'Tee+)

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

    # Drop rows that have too many nan
    df = df.loc[df[columns].isna().sum(axis=1) < 10]

    # Interpolate nan values
    df[columns] = interp_numba(df[columns].values)

    # Convert power to dB
    df[columns] = 10*np.log10(df[columns])
    return df

def assign_sharp_drop(df):
    df = df.copy()
    
    # Loop for each group (group by ext_act_id)
    list_df = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        # Find which "period" (1, 2, or 3) gives min(diff+backscatter)
        periods = int(np.argmin([
            (df_grp[columns_model].diff(periods=1, axis=1)+df_grp[columns_model]).min(axis=1).min(),
            (df_grp[columns_model].diff(periods=2, axis=1)+df_grp[columns_model]).min(axis=1).min(),
            (df_grp[columns_model].diff(periods=3, axis=1)+df_grp[columns_model]).min(axis=1).min()
        ])+1)
        
        # Find which column
        drop = df_grp[columns_model].diff(periods=periods, axis=1)
        coef = drop+df_grp[columns_model]
        flood_column = coef.min().idxmin()
        
        # Add drop value
        df_grp["drop"] = drop[flood_column]
        
        # Add the most extreme diff*backscatter
        df_grp["drop+bc"] = coef.min(axis=1).values
        
        # Add sharp-drop column
        flood_column = int(flood_column[1:])
        df_grp["drop_column"] = f"t{flood_column}"
        
        # Extract data (-2, +2)
        df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]] = df_grp[[f"t{i}" for i in range(flood_column-2, flood_column+3)]].values
        
        # Background columns (before flood)
        columns_background = [f"t{i}" for i in range(max(0, flood_column-10), flood_column-1)]
        df_grp["background_bc"] = df_grp[columns_background].median(axis=1).median()

        # Append to list
        list_df.append(df_grp)    
        
    # Concat and return
    df = pd.concat(list_df, ignore_index=True)
    return df
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"

# Define columns group
columns = [f"t{i}" for i in range(0, 30)]
columns_large = [f"t{i}" for i in range(0, 35)]
columns_age1 = [f"t{i}" for i in range(0, 7)]   # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)] # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)] # 120-179

columns_model = columns_age1[-1:]+columns_age2+columns_age3+columns_age4
#%%
strip_id = "304"

# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]

# Load df_vew
df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df["in_season_rice_f"] == 1]
df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]

# Down-Samping 
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] > 0, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] > 0)]

# Convert to db
df = convert_power_to_db(df, columns_large)

# Assign sharp drop
df = assign_sharp_drop(df)











