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
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210629"
# Define columns' group name
columns = [f"t{i}" for i in range(0, 30)]
columns_large = [f"t{i}" for i in range(0, 35)]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179
#%%
strip_id = "304"
os.makedirs(os.path.join(root_save, strip_id), exist_ok=True)

df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df["in_season_rice_f"] == 1]
df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
df = df.assign(loss_ratio_bin = np.digitize(df["loss_ratio"], [0, 0.25, 0.5, 0.75], right=True))
#%%
df = convert_power_to_db(df, columns_large)
#%%
for loss_ratio_bin in sorted(df["loss_ratio_bin"].unique()):
    print(loss_ratio_bin)
    list_ext_act_id = df.loc[df["loss_ratio_bin"] == loss_ratio_bin, "ext_act_id"].drop_duplicates().sample(n=200)
    df_grp = df[df["ext_act_id"].isin(list_ext_act_id)]
    for ext_act_id in list_ext_act_id:
        plt.close("all")
        fig, ax = initialize_plot()
        row = df.loc[df["ext_act_id"] == ext_act_id]
        n_pixels = len(row)
        row = row.iloc[0]
        ax.plot(df.loc[df["ext_act_id"] == ext_act_id, columns_large].T)
        ax.grid()
        ax.set_xlabel("Date")
        ax.set_ylabel("Backscatter Coefficient (dB)")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, ext_act_id:{int(row.ext_act_id)}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})\nLoss ratio:{row.loss_ratio:.2f} (Bin:{row.loss_ratio_bin})")
        if row.loss_ratio_bin != 0:
            ax.axvline((row.DANGER_DATE-row.final_plant_date).days//6, color="red", linestyle="--")
            
        # Save fig
        if n_pixels == 1:
            folder_save = os.path.join(root_save, strip_id, "small", str(row.loss_ratio_bin))
        elif (n_pixels > 1) and (n_pixels <= 5):
            folder_save = os.path.join(root_save, strip_id, "medium", str(row.loss_ratio_bin))
        elif n_pixels >5:
            folder_save = os.path.join(root_save, strip_id, "large", str(row.loss_ratio_bin))
        os.makedirs(folder_save, exist_ok=True)
        fig.savefig(os.path.join(folder_save, f"Bin{row.loss_ratio_bin}_{row.ext_act_id}.png"), bbox_inches="tight")
#%%









