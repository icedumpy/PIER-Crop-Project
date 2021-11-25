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

def get_temporal_file_info(root_temporal):
    list_info = []
    for file in os.listdir(root_temporal):
        list_info.append({
            "filename": file,
            "path": os.path.join(root_temporal, file),
            "p_code": file[-15:-13],
            "strip_id": file[-11:-8]
        })
    df_info = pd.DataFrame(list_info)
    return df_info

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

    # Drop row with mostly nan
    df = df.loc[df[columns].isna().sum(axis=1) < 10]

    # Interpolate nan values
    df[columns] = interp_numba(df[columns].values)

    # Convert power to dB
    df[columns] = 10*np.log10(df[columns])
    return df

def extract_features(df, columns_temporal):
    # Loop for each group (group by ext_act_id)
    list_data = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        # Get basic info
        df_grp_info = df_grp.iloc[0, 30:-2]
        
        # If too early (before July), skip first growth stage (first 40 days)
        if df_grp.iloc[0]["final_plant_date"].month < 7:
            columns = columns_temporal[6:]
        else:
            columns = columns_temporal
        
        # Find which "period" (1, 2, or 3) has min(diff+backscatter)
        periods = int(np.argmin([
            (df_grp[columns].diff(periods=1, axis=1)+df_grp[columns]).min(axis=1).min(),
            (df_grp[columns].diff(periods=2, axis=1)+df_grp[columns]).min(axis=1).min(),
            (df_grp[columns].diff(periods=3, axis=1)+df_grp[columns]).min(axis=1).min()
        ])+1)
        
        # Find which column
        drop = df_grp[columns].diff(periods=periods, axis=1)
        coef = drop+df_grp[columns]
        flood_column = int(coef.min().idxmin()[1:])
        
        # Get data (flood_column-lag -> flood_column-lag+8)
        columns_for_feature = [f"t{i}" for i in range(max(0, flood_column-periods), min(30, flood_column-periods+8))]
        
        # =============================================================================
        # Create dict of extracted features (plot-level)
        # =============================================================================
        dict_extracted_features = {
            **df_grp_info.to_dict(),
            "total_period": len(columns_for_feature)
        }
        # Thresholding range(-12, -19, -1)
        list_thresholds = list(range(-12, -19, -1))
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
            
            # Plot-level features (Strict)
            arr_consecutive_strict_grp = np.digitize(list_consecutive_strict, [1, 2, 3, 4])
            pct_plot_1_6_strict   = 100*(arr_consecutive_strict_grp == 1).sum()/len(arr_consecutive_strict_grp)
            pct_plot_7_12_strict  = 100*(arr_consecutive_strict_grp == 2).sum()/len(arr_consecutive_strict_grp)
            pct_plot_13_18_strict = 100*(arr_consecutive_strict_grp == 3).sum()/len(arr_consecutive_strict_grp)
            pct_plot_19_strict    = 100*(arr_consecutive_strict_grp == 4).sum()/len(arr_consecutive_strict_grp)

            # Plot-level features (Relax)
            arr_consecutive_relax_grp = np.digitize(list_consecutive_relax, [1, 2, 3, 4])
            pct_plot_1_6_relax   = 100*(arr_consecutive_relax_grp == 1).sum()/len(arr_consecutive_relax_grp)
            pct_plot_7_12_relax  = 100*(arr_consecutive_relax_grp == 2).sum()/len(arr_consecutive_relax_grp)
            pct_plot_13_18_relax = 100*(arr_consecutive_relax_grp == 3).sum()/len(arr_consecutive_relax_grp)
            pct_plot_19_relax    = 100*(arr_consecutive_relax_grp == 4).sum()/len(arr_consecutive_relax_grp)            
            
            # Create dict of new features -> Append to list
            dict_extracted_features = {
                **dict_extracted_features,
                **{ f"pct_of_plot_with_backscatter_under({threshold})_1-6_days_strict": pct_plot_1_6_strict,
                    f"pct_of_plot_with_backscatter_under({threshold})_7-12_days_strict": pct_plot_7_12_strict,
                    f"pct_of_plot_with_backscatter_under({threshold})_13-18_days_strict": pct_plot_13_18_strict,
                    f"pct_of_plot_with_backscatter_under({threshold})_19+_days_strict": pct_plot_19_strict,
                    f"pct_of_plot_with_backscatter_under({threshold})_1-6_days_relax": pct_plot_1_6_relax,
                    f"pct_of_plot_with_backscatter_under({threshold})_7-12_days_relax": pct_plot_7_12_relax,
                    f"pct_of_plot_with_backscatter_under({threshold})_13-18_days_relax": pct_plot_13_18_relax,
                    f"pct_of_plot_with_backscatter_under({threshold})_19+_days_relax": pct_plot_19_relax,
                  }
            }
        
        # After finish extracting features (every threshold)
        list_data.append(dict_extracted_features)
    
    df = pd.DataFrame(list_data)
    return df
#%%
root_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_features(same-idea-as-GISTDA-Flood)"
df_info = get_temporal_file_info(root_temporal)
columns_temporal = [f"t{i}" for i in range(30)]
#%%
for idx, file_info in df_info.iterrows():
    if idx%2 == 0:
        continue
    
    print(idx, file_info["filename"])
    path_save = os.path.join(root_save, file_info["filename"].replace("temporal", "features"))
    if os.path.exists(path_save):
        continue
    
    # Load dataframe
    df = pd.read_parquet(file_info["path"])
    df = df.drop(columns=['t30', 't31', 't32', 't33', 't34'])
    
    # Process data (Clean & Convert power to dB)
    df = df[df["in_season_rice_f"] == 1]
    df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]
    df = convert_power_to_db(df, columns_temporal)
    
    # Extract Features
    df = extract_features(df, columns_temporal)
    
    # Save file
    if len(df) != 0:
        df.to_parquet(path_save)
#%%
#%%
# df_grp = pd.DataFrame(
#     np.array([[-20, -20, -20, -20, -20, -20, -20, -20, ], 
#               [-19, -18, -17, -16, -16, -15, -17, -18, ]]),
#     # columns = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]
#     columns = columns_for_feature
# )       








