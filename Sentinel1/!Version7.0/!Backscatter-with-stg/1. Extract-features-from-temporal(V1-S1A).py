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

    # Drop row with mostly nan
    df = df.loc[df[columns].isna().sum(axis=1) < 10]

    # Interpolate nan values
    df[columns] = interp_numba(df[columns].values)

    # Convert power to dB
    df[columns] = 10*np.log10(df[columns])
    return df

def initialize_plot(ylim=(-20, 0)):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw group age
    ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax.axvspan(6.5, 15, alpha=0.2, color='green')
    ax.axvspan(15.0, 20, alpha=0.2, color='yellow')
    ax.axvspan(20.0, 34, alpha=0.2, color='purple')

    [ax.axvline(i, color="black") for i in [6.5, 15, 20]]

    # Add group descriptions
    ax.text(3, ylim[-1]+0.25, "0-40 days", horizontalalignment="center")
    ax.text(10.5, ylim[-1]+0.25, "40-90 days", horizontalalignment="center")
    ax.text(17.5, ylim[-1]+0.25, "90-120 days", horizontalalignment="center")
    ax.text(27.5, ylim[-1]+0.25, "120+ days", horizontalalignment="center")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(35))
    ax.set_yticks(np.arange(*ylim))

    return fig, ax

def assign_sharp_drop(df, columns_stg, label):
    df = df.copy()
    # Temporary column (t15)
    df["t15"] = df["t14"]
    
    # Loop for each group (group by ext_act_id)
    list_df = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        # Find which "period" (1 or 2) gives min(diff+backscatter)
        periods = int(np.argmin([
            (df_grp[columns].diff(periods=1, axis=1)+df_grp[columns]).min(axis=1).min(),
            (df_grp[columns].diff(periods=2, axis=1)+df_grp[columns]).min(axis=1).min(),
        ])+1)
        
        # Find which column
        drop = df_grp[columns_stg].diff(periods=periods, axis=1)
        coef = drop+df_grp[columns_stg]
        flood_column = coef.min().idxmin()
        
        # Add drop value
        df_grp[f"drop_{label}"] = drop[flood_column]
        
        # Add the most extreme diff*backscatter
        df_grp[f"drop+bc_{label}"] = coef.min(axis=1).values
        
        # Add sharp-drop column
        flood_column = int(flood_column[1:])
        df_grp[f"drop_column_{label}"] = f"t{flood_column}"
        
        # Extract data (-2, +2)
        df_grp[[f"bc(t-2)_{label}", f"bc(t-1)_{label}", f"bc(t)_{label}", f"bc(t+1)_{label}", f"bc(t+2)_{label}"]] = df_grp[[f"t{i}" for i in range(flood_column-2, flood_column+3)]].values
        
        # Background columns (before flood)
        if columns_stg == ['t0', 't1', 't2', 't3', 't4', 't5', 't6']:
            columns_background = [f"t{i}" for i in range(0 , 7 )]
        else:
            columns_background = [f"t{i}" for i in range(max(0, flood_column-10), flood_column-1)]
        df_grp[f"background_bc_{label}"] = df_grp[columns_background].median(axis=1)

        # Background - bc
        df_grp[f"background_bc-bc(t)_{label}"] = df_grp[f"background_bc_{label}"]-df_grp[f"bc(t)_{label}"]

        # Append to list
        list_df.append(df_grp)
        
    # Concat and return
    df = pd.concat(list_df, ignore_index=True)
    return df

def convert_pixel_level_to_plot_level(df):
    df = df.copy()
    
    list_dict_plot = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        loss_ratio = df_grp.iloc[0]["loss_ratio"]
        PLANT_PROVINCE_CODE = df_grp.iloc[0]["PLANT_PROVINCE_CODE"]
        PLANT_AMPHUR_CODE = df_grp.iloc[0]["PLANT_AMPHUR_CODE"]
        PLANT_TAMBON_CODE = df_grp.iloc[0]["PLANT_TAMBON_CODE"]
        DANGER_TYPE = df_grp.iloc[0]["DANGER_TYPE"]
        BREED_CODE = df_grp.iloc[0]['BREED_CODE']
        n_pixel = len(df_grp)
        dict_plot = {
            "ext_act_id":ext_act_id,
            'PLANT_PROVINCE_CODE':PLANT_PROVINCE_CODE,
            'PLANT_AMPHUR_CODE':PLANT_AMPHUR_CODE,
            'PLANT_TAMBON_CODE':PLANT_TAMBON_CODE,
            'DANGER_TYPE':DANGER_TYPE,
            'BREED_CODE':BREED_CODE,
            "loss_ratio":loss_ratio,
            "n_pixel":n_pixel,
        }
        for stg in ["nrt_stg1", "nrt_stg2", "nrt_stg3", "nrt_stg4"]:
            # Agg parameters
            drop_min = df_grp[f"drop_{stg}"].min()
            drop_max = df_grp[f"drop_{stg}"].max()
            drop_p25 = df_grp[f"drop_{stg}"].quantile(0.25)
            drop_p50 = df_grp[f"drop_{stg}"].quantile(0.50)
            drop_p75 = df_grp[f"drop_{stg}"].quantile(0.75)
            drop_bc_min = df_grp[f"drop+bc_{stg}"].min()
            drop_bc_max = df_grp[f"drop+bc_{stg}"].max()
            drop_bc_p25 = df_grp[f"drop+bc_{stg}"].quantile(0.25)
            drop_bc_p50 = df_grp[f"drop+bc_{stg}"].quantile(0.50)
            drop_bc_p75 = df_grp[f"drop+bc_{stg}"].quantile(0.75)
            bc_min = df_grp[[f"bc(t-2)_{stg}", f"bc(t-1)_{stg}", f"bc(t)_{stg}", f"bc(t+1)_{stg}", f"bc(t+2)_{stg}"]].min(axis=0).values
            bc_max = df_grp[[f"bc(t-2)_{stg}", f"bc(t-1)_{stg}", f"bc(t)_{stg}", f"bc(t+1)_{stg}", f"bc(t+2)_{stg}"]].max(axis=0).values
            bc_p25 = df_grp[[f"bc(t-2)_{stg}", f"bc(t-1)_{stg}", f"bc(t)_{stg}", f"bc(t+1)_{stg}", f"bc(t+2)_{stg}"]].quantile(0.25, axis=0).values
            bc_p50 = df_grp[[f"bc(t-2)_{stg}", f"bc(t-1)_{stg}", f"bc(t)_{stg}", f"bc(t+1)_{stg}", f"bc(t+2)_{stg}"]].quantile(0.50, axis=0).values
            bc_p75 = df_grp[[f"bc(t-2)_{stg}", f"bc(t-1)_{stg}", f"bc(t)_{stg}", f"bc(t+1)_{stg}", f"bc(t+2)_{stg}"]].quantile(0.75, axis=0).values
            background_bc_min = df_grp[f"background_bc_{stg}"].min()
            background_bc_max = df_grp[f"background_bc_{stg}"].max()
            background_bc_p25 = df_grp[f"background_bc_{stg}"].quantile(0.25)
            background_bc_p50 = df_grp[f"background_bc_{stg}"].quantile(0.50)
            background_bc_p75 = df_grp[f"background_bc_{stg}"].quantile(0.75)
            background_bc_minus_bc_t_min = df_grp[f"background_bc-bc(t)_{stg}"].min()
            background_bc_minus_bc_t_max = df_grp[f"background_bc-bc(t)_{stg}"].max()
            background_bc_minus_bc_t_p25 = df_grp[f"background_bc-bc(t)_{stg}"].quantile(0.25)
            background_bc_minus_bc_t_p50 = df_grp[f"background_bc-bc(t)_{stg}"].quantile(0.50)
            background_bc_minus_bc_t_p75 = df_grp[f"background_bc-bc(t)_{stg}"].quantile(0.75)
            
            drop_column = df_grp.iloc[0][f"drop_column_{stg}"]
            if drop_column in columns_stg1:
                drop_age = 1
            elif drop_column in columns_stg2:
                drop_age = 2
            elif drop_column in columns_stg3:
                drop_age = 3
            elif drop_column in columns_stg4:
                drop_age = 4

            # Create dict of parameters
            dict_plot = {**dict_plot, 
                f"drop_age_{stg}":drop_age,
                f"drop_min_{stg}":drop_min,
                f"drop_max_{stg}":drop_max,
                f"drop_p25_{stg}":drop_p25,
                f"drop_p50_{stg}":drop_p50,
                f"drop_p75_{stg}":drop_p75,
                f"drop+bc_min_{stg}":drop_bc_min,
                f"drop+bc_max_{stg}":drop_bc_max,
                f"drop+bc_p25_{stg}":drop_bc_p25,
                f"drop+bc_p50_{stg}":drop_bc_p50,
                f"drop+bc_p75_{stg}":drop_bc_p75,
                f"bc(t-2)_min_{stg}":bc_min[0],
                f"bc(t-1)_min_{stg}":bc_min[1],
                f"bc(t)_min_{stg}"  :bc_min[2],
                f"bc(t+1)_min_{stg}":bc_min[3],
                f"bc(t+2)_min_{stg}":bc_min[4],
                f"bc(t-2)_max_{stg}":bc_max[0],
                f"bc(t-1)_max_{stg}":bc_max[1],
                f"bc(t)_max_{stg}"  :bc_max[2],
                f"bc(t+1)_max_{stg}":bc_max[3],
                f"bc(t+2)_max_{stg}":bc_max[4],
                f"bc(t-2)_p25_{stg}":bc_p25[0],
                f"bc(t-1)_p25_{stg}":bc_p25[1],
                f"bc(t)_p25_{stg}"  :bc_p25[2],
                f"bc(t+1)_p25_{stg}":bc_p25[3],
                f"bc(t+2)_p25_{stg}":bc_p25[4],
                f"bc(t-2)_p50_{stg}":bc_p50[0],
                f"bc(t-1)_p50_{stg}":bc_p50[1],
                f"bc(t)_p50_{stg}"  :bc_p50[2],
                f"bc(t+1)_p50_{stg}":bc_p50[3],
                f"bc(t+2)_p50_{stg}":bc_p50[4],
                f"bc(t-2)_p75_{stg}":bc_p75[0],
                f"bc(t-1)_p75_{stg}":bc_p75[1],        
                f"bc(t)_p75_{stg}"  :bc_p75[2],
                f"bc(t+1)_p75_{stg}":bc_p75[3],
                f"bc(t+2)_p75_{stg}":bc_p75[4],
                f"background_bc_min_{stg}":background_bc_min,
                f"background_bc_max_{stg}":background_bc_max,
                f"background_bc_p25_{stg}":background_bc_p25,
                f"background_bc_p50_{stg}":background_bc_p50,
                f"background_bc_p75_{stg}":background_bc_p75,
                f"background_bc_minus_bc_t_min_{stg}" : background_bc_minus_bc_t_min,
                f"background_bc_minus_bc_t_max_{stg}" : background_bc_minus_bc_t_max,
                f"background_bc_minus_bc_t_p25_{stg}" : background_bc_minus_bc_t_p25,
                f"background_bc_minus_bc_t_p50_{stg}" : background_bc_minus_bc_t_p50,
                f"background_bc_minus_bc_t_p75_{stg}" : background_bc_minus_bc_t_p75,
            }
        # Append dict to list
        list_dict_plot.append(dict_plot)
        
    # Create dataframe and return
    df_plot = pd.DataFrame(list_dict_plot)
    return df_plot

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1a_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1a_vew_plant_info_official_polygon_disaster_all_rice_by_year_version_for_nrt(at-False)"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]

os.makedirs(root_save, exist_ok=True)
#%%
# Define columns for each growth stage
columns = [f"t{i}" for i in range(15)]
columns_stg1 = [f"t{i}" for i in range(0, 3)]   # 0-35
columns_stg2 = [f"t{i}" for i in range(3, 8)]   # 36-95
columns_stg3 = [f"t{i}" for i in range(8, 10)]  # 96-119
columns_stg4 = [f"t{i}" for i in range(10, 15)] # 120-179
#%%
# (vew == temporal) by the way.
for file in os.listdir(root_vew):
    print(file)
    path_file = os.path.join(root_vew, file)
    path_save = os.path.join(root_save, file.replace("temporal", "version_for_nrt"))
    
    # Load data
    df = pd.read_parquet(path_file)
    df = pd.merge(df, df_rice_code, on="BREED_CODE", how="left")

    # Some cleaning
    df = df[df["in_season_rice_f"] == 1]
    df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]
    df = convert_power_to_db(df, columns)
    
    # Find shard drop for 
    df = assign_sharp_drop(df, columns_stg=columns_stg1[1:], label="nrt_stg1") # Have to skip "t0"
    df = assign_sharp_drop(df, columns_stg=columns_stg1[1:]+columns_stg2, label="nrt_stg2") # Have to skip "t0"
    df = assign_sharp_drop(df, columns_stg=columns_stg1[1:]+columns_stg2+columns_stg3, label="nrt_stg3") # Have to skip "t0"
    df = assign_sharp_drop(df, columns_stg=columns_stg1[1:]+columns_stg2+columns_stg3+columns_stg4, label="nrt_stg4") # Have to skip "t0"
    
    # Convert from pixel-level to plot-level
    df_plot = convert_pixel_level_to_plot_level(df)
    
    # Save file
    df_plot.to_parquet(path_save)
#%%

