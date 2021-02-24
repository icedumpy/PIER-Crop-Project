import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
#%%
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

    # Assign label
    df.loc[df["loss_ratio"] == 0, "label"] = 0
    df.loc[df["loss_ratio"] >= 0.8, "label"] = 1
    return df

def initialize_plot(mean_normal, ci_normal, ylim=(-20, -5)):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot normal curve
    ax.plot(mean_normal, linestyle="--", marker="o", color="red", label="Mean(nonFlood) w/ 90% CI")

    # Plot confidence interval
    ax.fill_between(range(31), (mean_normal-ci_normal), (mean_normal+ci_normal), color='red', alpha=.5)

    # Draw group age
    ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax.axvspan(6.5, 15.5, alpha=0.2, color='green')
    ax.axvspan(15.5, 30, alpha=0.2, color='yellow')
    [ax.axvline(i, color="black") for i in [6.5, 15.5]]
    
    # Add group descriptions
    ax.text(2.5, ylim[-1]+0.25, "0-40 days")
    ax.text(10, ylim[-1]+0.25, "40-90 days")
    ax.text(22, ylim[-1]+0.25, "90+ days")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(31))
    ax.set_yticks(np.arange(*ylim))

    return fig, ax
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210224\Fig"
strip_id = "304"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
columns = df.columns[:31]
columns_age1 = [f"t{i}" for i in range(0, 7)]
columns_age2 = [f"t{i}" for i in range(7, 16)]
columns_age3 = [f"t{i}" for i in range(16, 31)]

# Convert power to db
df = convert_power_to_db(df, columns)

# Assign mean
df = df.assign(mean = df[columns].mean(axis=1))
df = df.assign(median = df[columns].median(axis=1))

# Diff
# for i in range(len(columns)-2):
#     df = df.assign(**{f"t{i+2}-t{i}":df[columns[i+2]]-df[columns[i]],
#                       f"t{i+1}-t{i}":df[columns[i+1]]-df[columns[i]]})

df_nonflood = df[df["label"] == 0]
df_flood = df[df["label"] == 1]

# Add flood column
df_flood = df_flood.assign(flood_column = (df_flood["START_DATE"]-df_flood["final_plant_date"]).dt.days//6)
#%%
plt.close("all")
fig, ax = plt.subplots(figsize=(16, 9))
























#%%
