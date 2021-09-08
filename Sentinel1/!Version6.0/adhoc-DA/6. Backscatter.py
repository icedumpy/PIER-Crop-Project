import os
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
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

    # Assign label
    df.loc[df["loss_ratio"] == 0, "label"] = 0
    df.loc[df["loss_ratio"] >  0, "label"] = 1
    df["label"] = df["label"].astype("uint8")
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
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
columns = [f"t{i}" for i in range(30)]
columns_large = [f"t{i}" for i in range(0, 35)]
#%%
for strip_id in ["304", "402"]:
    df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    df = df[df["in_season_rice_f"] == 1]
    df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
    df = convert_power_to_db(df, columns_large)
    break
#%%
df.loc[df["label"] == 1, "danger_column"] = "t"+((df.loc[df["label"] == 1, "DANGER_DATE"]-df.loc[df["label"] == 1, "final_plant_date"]).dt.days//6).astype(str)
#%%

#%%
fig, ax = initialize_plot()
df.loc[df["label"] == 0, columns_large].mean(axis=0).plot()
df.loc[df["label"] == 1, columns_large].mean(axis=0).plot()
#%%
for danger_column, df_grp in df.groupby("danger_column"):
    print(danger_column)




















