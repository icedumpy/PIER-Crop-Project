import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from icedumpy.df_tools import set_index_for_loc
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
    ax.plot(mean_normal, linestyle="--", marker="o", color="green", label="Mean (nonFlood) w/ 90% CI")

    # Plot confidence interval
    ax.fill_between(range(30), (mean_normal-ci_normal), (mean_normal+ci_normal), color='green', alpha=.2)

    # Draw group age
    ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax.axvspan(6.5, 15, alpha=0.2, color='green')
    ax.axvspan(15, 20, alpha=0.2, color='yellow')
    ax.axvspan(20, 29, alpha=0.2, color='purple')
    [ax.axvline(i, color="black") for i in [6.5, 15.0, 20]]
    
    # Add group descriptions
    ax.text(3.0, ylim[-1]+0.25, "0-40 days", horizontalalignment="center")
    ax.text(11, ylim[-1]+0.25, "40-90 days", horizontalalignment="center")
    ax.text(17.5, ylim[-1]+0.25, "90-120 days", horizontalalignment="center")
    ax.text(25, ylim[-1]+0.25, "120+ days", horizontalalignment="center")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(30))
    ax.set_yticks(np.arange(*ylim))
    
    # Change x label name
    ax.set_xticklabels([f"{6*i}-{6*(i+1)-1}" for i in range(30)], rotation="90")
    return fig, ax
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
strip_id = "304"
#%%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    df = df.drop(columns="t30") # Column "t30" is already out of season
    df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]

    columns = df.columns[:30]
    columns_age1 = [f"t{i}" for i in range(0, 7)] # 0-41
    columns_age2 = [f"t{i}" for i in range(7, 15)] # 42-89
    columns_age3 = [f"t{i}" for i in range(15, 20)] # 90-119
    columns_age4 = [f"t{i}" for i in range(20, 30)] # 120-179
    
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
                      "mean(age4)":df[columns_age4].mean(axis=1),
                      "median(age4)":df[columns_age4].median(axis=1),
                      })
    df = df.assign(**{"median-min(age1)":df["median(age1)"]-df[columns_age1].min(axis=1),
                      "median-min(age2)":df["median(age2)"]-df[columns_age2].min(axis=1),
                      "median-min(age3)":df["median(age3)"]-df[columns_age3].min(axis=1),
                      "median-min(age4)":df["median(age4)"]-df[columns_age4].min(axis=1),
                      })
    
    # Assign sum of value under median (instead of area under median)
    df = df.assign(**{"area-under-median(age1)":(-df[columns_age1].sub(df["median(age1)"], axis=0)).clip(lower=0, upper=None).sum(axis=1),
                      "area-under-median(age2)":(-df[columns_age2].sub(df["median(age2)"], axis=0)).clip(lower=0, upper=None).sum(axis=1),
                      "area-under-median(age3)":(-df[columns_age3].sub(df["median(age3)"], axis=0)).clip(lower=0, upper=None).sum(axis=1),
                      "area-under-median(age4)":(-df[columns_age4].sub(df["median(age4)"], axis=0)).clip(lower=0, upper=None).sum(axis=1)
                      })
    
    df_nonflood = df[df["label"] == 0]
    df_flood = df[df["label"] == 1]
    
    # Add flood column
    df_flood = df_flood.assign(flood_column = (df_flood["START_DATE"]-df_flood["final_plant_date"]).dt.days//6)
    
    # Calculate confidence interval
    mean_normal = df_nonflood[columns].mean(axis=0).values
    ci_normal = 1.645*df_nonflood[columns].std(axis=0).values/mean_normal
    
    df_sample = df_nonflood.groupby(["ext_act_id"]).sample(n=1).head(20)
    for i, (index, row) in enumerate(df_sample.iterrows()):
        plt.close("all")
        fig, ax = initialize_plot(mean_normal, ci_normal, ylim=(-20, 0))
        
        # Plot temporal
        ax.plot(row[columns].values, linestyle="--", marker='o', color="blue", label="Sample(nonFlood)")
        
        # Plot mean, median (age1)
        ax.hlines(row["median(age1)"], xmin=0, xmax=6.5, linestyle="--", linewidth=2.5, color="orange", label="Median (Age1)")
        
        # Plot mean, median (age2)
        ax.hlines(row["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--", linewidth=2.5, color="gray", label="Median (age2)")
        
        # Plot mean, median (age3)
        ax.hlines(row["median(age3)"], xmin=15.0, xmax=20, linestyle="--", linewidth=2.5, color="purple", label="Median (age3)")
        
        # Plot mean, median (age4)
        ax.hlines(row["median(age4)"], xmin=20.0, xmax=29, linestyle="--", linewidth=2.5, color="yellow", label="Median (age4)")
        
        # Add final details
        ax.legend(loc=4)
        ax.grid(linestyle="--")
        ax.set_xlabel("Rice age (day)")
        ax.set_ylabel("Backscatter coefficient (dB)")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{row.ext_act_id}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})")
        
        # Savefig
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210228\normal", f"{strip_id}_{i+1}.png"), bbox_inches="tight")
    
    df_sample = df_flood.groupby(["ext_act_id"]).sample(n=1).head(20)
    for i, (index, row) in enumerate(df_sample.iterrows()):
        plt.close("all")
        fig, ax = initialize_plot(mean_normal, ci_normal, ylim=(-20, 0))
        
        # Plot temporal
        ax.plot(row[columns].values, linestyle="--", marker='o', color="blue", label="Sample(Flood)")
        
        # Plot mean, median (age1)
        ax.hlines(row["median(age1)"], xmin=0, xmax=6.5, linestyle="--", linewidth=2.5, color="orange", label="Median (Age1)")
        
        # Plot mean, median (age2)
        ax.hlines(row["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--", linewidth=2.5, color="gray", label="Median (age2)")
        
        # Plot mean, median (age3)
        ax.hlines(row["median(age3)"], xmin=15.0, xmax=29, linestyle="--", linewidth=2.5, color="purple", label="Median (age3)")

        # Plot mean, median (age4)
        ax.hlines(row["median(age4)"], xmin=20.0, xmax=29, linestyle="--", linewidth=2.5, color="yellow", label="Median (age4)")
                
        # Draw start date
        ax.axvline(row.flood_column, color="red", linestyle="--")
        ax.text(row.flood_column, ax.get_ylim()[1]+0.8, "Reported flood date", horizontalalignment="center", color="red")
        
        # Add final details
        ax.legend(loc=4)
        ax.grid(linestyle="--")
        ax.set_xlabel("Rice age (day)")
        ax.set_ylabel("Backscatter coefficient (dB)")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{row.ext_act_id}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})")
        
        # Savefig
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210228\flood", f"{strip_id}_{i+1}.png"), bbox_inches="tight")
#%%
