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
    ax.plot(mean_normal, linestyle="--", marker="o", color="red", label="Mean (nonFlood) w/ 90% CI")

    # Plot confidence interval
    ax.fill_between(range(30), (mean_normal-ci_normal), (mean_normal+ci_normal), color='red', alpha=.3)

    # Draw group age
    ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax.axvspan(6.5, 15.0, alpha=0.2, color='green')
    ax.axvspan(15.0, 29, alpha=0.2, color='yellow')
    [ax.axvline(i, color="black") for i in [6.5, 15.0]]
    
    # Add group descriptions
    ax.text(2.5, ylim[-1]+0.25, "0-40 days")
    ax.text(10, ylim[-1]+0.25, "40-90 days")
    ax.text(22, ylim[-1]+0.25, "90+ days")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(30))
    ax.set_yticks(np.arange(*ylim))
    
    # Change x label name
    ax.set_xticklabels([f"{6*i}-{6*(i+1)}" for i in range(30)], rotation="90")
    return fig, ax
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210224\Fig"
strip_id = "304"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df.drop(columns="t30") # Column "t30" is already out of season
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
columns = df.columns[:30]
columns_age1 = [f"t{i}" for i in range(0, 7)]
columns_age2 = [f"t{i}" for i in range(7, 15)]
columns_age3 = [f"t{i}" for i in range(15, 30)]

# Convert power to db
df = convert_power_to_db(df, columns)

# Assign mean, median (overall)
df = df.assign(**{"mean":df[columns].mean(axis=1),
                  "median":df[columns].median(axis=1),
                  "mean(age1)":df[columns_age1].mean(axis=1),
                  "median(age1)":df[columns_age1].median(axis=1),
                  "mean(age2)":df[columns_age2].mean(axis=1),
                  "median(age2)":df[columns_age2].median(axis=1),
                  "mean(age3)":df[columns_age3].mean(axis=1),
                  "median(age3)":df[columns_age3].median(axis=1),
                  })
df = df.assign(**{"median-min(age1)":df["median(age1)"]-df[columns_age1].min(axis=1),
                  "median-min(age2)":df["median(age2)"]-df[columns_age2].min(axis=1),
                  "median-min(age3)":df["median(age3)"]-df[columns_age3].min(axis=1),
                  })

# Assign sum of value under median (instead of area under median)
df = df.assign(**{"area-under-median(age1)":(-df[columns_age1].sub(df["median(age1)"], axis=0)).clip(lower=0, upper=None).sum(axis=1),
                  "area-under-median(age2)":(-df[columns_age2].sub(df["median(age2)"], axis=0)).clip(lower=0, upper=None).sum(axis=1),
                  "area-under-median(age3)":(-df[columns_age3].sub(df["median(age3)"], axis=0)).clip(lower=0, upper=None).sum(axis=1)})

df_nonflood = df[df["label"] == 0]
df_flood = df[df["label"] == 1]

# Add flood column
df_flood = df_flood.assign(flood_column = (df_flood["START_DATE"]-df_flood["final_plant_date"]).dt.days//6)

# Calculate confidence interval
mean_normal = df_nonflood[columns].mean(axis=0).values
ci_normal = 1.645*df_nonflood[columns].std(axis=0).values/mean_normal
#%%
df_sample = df.sample(n=1).squeeze()

plt.close("all")
fig, ax = initialize_plot(mean_normal, ci_normal, ylim=(-20, 0))

# Plot temporal
ax.plot(df_sample[columns].values, linestyle="--", marker='o', color="blue", label="Flood")

# Plot mean, median (age1)
ax.hlines(df_sample["median(age1)"], xmin=0, xmax=6.5, linestyle="--", color="orange", label="Median (Age1)")

# Plot mean, median (age2)
ax.hlines(df_sample["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--", color="gray", label="Median (age2)")

# Plot mean, median (age3)
ax.hlines(df_sample["median(age3)"], xmin=15.0, xmax=29, linestyle="--", color="purple", label="Median (age3)")

# Add final details
ax.legend(loc=4)
ax.grid(linestyle="--")
ax.set_xlabel("Rice age (day)")
ax.set_ylabel("Backscatter coefficient (dB)")
#%%

