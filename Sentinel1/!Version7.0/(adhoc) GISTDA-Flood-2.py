import os
import datetime
import rasterio
import numpy as np
import pandas as pd
from numba import jit
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
    ax.axvspan(20.0, 29, alpha=0.2, color='purple')

    [ax.axvline(i, color="black") for i in [6.5, 15, 20]]

    # Add group descriptions
    ax.text(3, ylim[-1]+0.25, "0-40 days", horizontalalignment="center")
    ax.text(10.5, ylim[-1]+0.25, "40-90 days", horizontalalignment="center")
    ax.text(17.5, ylim[-1]+0.25, "90-120 days", horizontalalignment="center")
    ax.text(25.5, ylim[-1]+0.25, "120+ days", horizontalalignment="center")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(30))
    ax.set_yticks(np.arange(*ylim))

    return fig, ax
#%%
root_df = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_gistda = r"F:\CROP-PIER\CROP-WORK\GISTDA-Flood\Rasterized"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211019\Fig"
columns = [f"t{i}" for i in range(0, 30)]
#%%
strip_id = "304"
dict_gistda = dict()
for file in [file for file in os.listdir(root_gistda) if file.split(".")[0][-3:] == strip_id]:
    dict_gistda["_".join(file.split("_")[:2])] = rasterio.open(os.path.join(root_gistda, file)).read()[0]
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df, file)) for file in os.listdir(root_df) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df["DANGER_TYPE"] == "อุทกภัย"]
df = convert_power_to_db(df, columns)
#%%
for index, df_row in df.iterrows():
    plant_date = df_row.final_plant_date
    harvt_date = df_row.final_plant_date+datetime.timedelta(days=180)
    
    plt.close("all")
    fig, ax = initialize_plot()
    ax.plot(df_row[columns].values)
    
    # Draw if GISTDA is flood
    plant_date = df_row.final_plant_date
    harvt_date = df_row.final_plant_date+datetime.timedelta(days=180)
    row = df_row["row"]
    col = df_row["col"]
    for key in dict_gistda.keys():
        start = datetime.datetime.strptime(key[:8], "%Y%m%d")
        if (plant_date <= start) and (start <= harvt_date):
            if dict_gistda[key][row, col] == 1:
                ax.axvspan(
                    ((start-plant_date)/6)/datetime.timedelta(days=1), 
                    ((start-plant_date)/6)/datetime.timedelta(days=1)+datetime.timedelta(days=5)/datetime.timedelta(days=6),
                    alpha=0.3, color='blue'
               )
                
    # Draw reported flood date
    ax.axvline(((df_row.DANGER_DATE-plant_date)//6).days, linestyle="--", color="red")
    ax.text(((df_row.DANGER_DATE-plant_date)//6).days, 0.1, "Reported", horizontalalignment="center", color="red")
    fig.savefig(os.path.join(root_save, f"{index}.png"), bbox_inches="tight")
#%%
index = 1141034
df_row = df.loc[index]
plant_date = df_row.final_plant_date
harvt_date = df_row.final_plant_date+datetime.timedelta(days=180)

plt.close("all")
fig, ax = initialize_plot()
ax.plot(df_row[columns].values)

# Draw if GISTDA is flood
plant_date = df_row.final_plant_date
harvt_date = df_row.final_plant_date+datetime.timedelta(days=180)
row = df_row["row"]
col = df_row["col"]
for key in dict_gistda.keys():
    start = datetime.datetime.strptime(key[:8], "%Y%m%d")
    if (plant_date <= start) and (start <= harvt_date):
        print(key)
        if dict_gistda[key][row, col] == 1:
            ax.axvspan(
                ((start-plant_date)/6)/datetime.timedelta(days=1), 
                ((start-plant_date)/6)/datetime.timedelta(days=1)+datetime.timedelta(days=5)/datetime.timedelta(days=6),
                alpha=0.3, color='blue'
            )
            print(((start-plant_date)/6)/datetime.timedelta(days=1), ((start-plant_date)/6)/datetime.timedelta(days=1)+datetime.timedelta(days=5)/datetime.timedelta(days=6),)

# Draw reported flood date
ax.axvline(((df_row.DANGER_DATE-plant_date)//6).days, linestyle="--", color="red")
ax.text(((df_row.DANGER_DATE-plant_date)//6).days, 0.1, "Reported", horizontalalignment="center", color="red")