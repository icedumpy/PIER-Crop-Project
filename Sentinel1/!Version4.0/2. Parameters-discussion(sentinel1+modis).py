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
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    # Plot normal curve
    ax[0].plot(mean_normal, linestyle="--", marker="o", color="green", label="Mean (nonFlood) w/ 90% CI")

    # Plot confidence interval
    ax[0].fill_between(range(30), (mean_normal-ci_normal), (mean_normal+ci_normal), color='green', alpha=.2)

    # Draw group age
    ax[0].axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax[0].axvspan(6.5, 15.0, alpha=0.2, color='green')
    ax[0].axvspan(15.0, 29, alpha=0.2, color='yellow')
    [ax[0].axvline(i, color="black") for i in [6.5, 15.0]]
    ax[1].axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax[1].axvspan(6.5, 15.0, alpha=0.2, color='green')
    ax[1].axvspan(15.0, 29, alpha=0.2, color='yellow')
    [ax[1].axvline(i, color="black") for i in [6.5, 15.0]]
    
    # Add group descriptions
    ax[0].text(2.5, ylim[-1]+0.25, "0-40 days")
    ax[0].text(10, ylim[-1]+0.25, "40-90 days")
    ax[0].text(22, ylim[-1]+0.25, "90+ days")

    # Set y limits
    ax[0].set_ylim(ylim)
    ax[1].set_ylim((-0.6, 0.6))

    # Add more ticks
    ax[0].set_xticks(range(30))
    ax[0].set_yticks(np.arange(*ylim))
    ax[1].set_xticks(range(0, 30, 2))
    ax[1].set_yticks(np.arange(-0.4, 0.6, 0.1))
    
    # Change x label name
    ax[0].set_xticklabels([f"{6*i}-{6*(i+1)-1}" for i in range(30)], rotation="90")
    ax[1].set_xticklabels([f"{6*i}-{6*(i+1)-1}" for i in range(0, 30, 2)], rotation="90")
    return fig, ax
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_modis_mapping = r"F:\CROP-PIER\CROP-WORK\modis_dataframe\dataset\2015-2019\mapping_ext_act_id_with_modis_pixel"
root_df_modis_ndwi = r"F:\CROP-PIER\CROP-WORK\modis_dataframe\dataset\2015-2019\ndwi-pixel-values-province-t14"
strip_id = "304"
#%%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    df = df.drop(columns="t30") # Column "t30" is already out of season
    df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
    
    # Load df_modis
    list_p = df["PLANT_PROVINCE_CODE"].astype(str).unique().tolist()
    df_mapping_modis = pd.concat([pd.read_parquet(os.path.join(root_df_modis_mapping, file)) for file in os.listdir(root_df_modis_mapping) if file.split(".")[0][-2:] in list_p], ignore_index=True)
    df_mapping_modis["row_col"] = df_mapping_modis["modis_pixel_row"].astype(str) + "-" + df_mapping_modis["modis_pixel_col"].astype(str)
    
    df_ndwi_modis = pd.concat([pd.read_parquet(os.path.join(root_df_modis_ndwi, file)) for file in os.listdir(root_df_modis_ndwi) if file.split(".")[0][-2:] in list_p], ignore_index=True)
    df_ndwi_modis = df_ndwi_modis.loc[df_ndwi_modis["year"].isin([2018, 2019])]
    
    df_mapping_modis = df_mapping_modis.loc[df_mapping_modis["row_col"].isin(df_ndwi_modis["row_col"])]
    
    # Drop somes
    df = df[df.ext_act_id.isin(df_mapping_modis.ext_act_id)]
    
    columns = df.columns[:30]
    columns_age1 = [f"t{i}" for i in range(0, 7)]
    columns_age2 = [f"t{i}" for i in range(7, 15)]
    columns_age3 = [f"t{i}" for i in range(15, 30)]
    columns_modis = [f"ndwi_t{i}" for i in range(1, 15)]
    
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
    
    count = 0
    df_sample = df_nonflood.groupby(["ext_act_id"]).sample(n=1).head(50)
    for index, row in df_sample.iterrows():
        modis = df_ndwi_modis[df_ndwi_modis["row_col"] == df_mapping_modis.loc[df_mapping_modis["ext_act_id"] == row.ext_act_id, "row_col"].values[0]]
        modis = modis[modis.prov_cd == row.PLANT_PROVINCE_CODE]
        modis = modis[modis["year"] == row.final_plant_date.year].squeeze()
        if len(modis) == 0:
            continue

        plt.close("all")
        fig, ax = initialize_plot(mean_normal, ci_normal, ylim=(-20, 0))
        
        # Plot temporal
        ax[0].plot(row[columns].values, linestyle="--", marker='o', color="blue", label="Sample(nonFlood)")
        
        # # Plot mean, median (age1)
        # ax[0].hlines(row["median(age1)"], xmin=0, xmax=6.5, linestyle="--", linewidth=2.5, color="orange", label="Median (Age1)")
        
        # # Plot mean, median (age2)
        # ax[0].hlines(row["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--", linewidth=2.5, color="gray", label="Median (age2)")
        
        # # Plot mean, median (age3)
        # ax[0].hlines(row["median(age3)"], xmin=15.0, xmax=29, linestyle="--", linewidth=2.5, color="purple", label="Median (age3)")
        
        # Add modis
        ax[1].plot([2*i-1 for i in range(1, 15)], modis[columns_modis].values.reshape(-1), linestyle="--", marker='o', color="blue", label="Modis-ndwi")
        ax[1].legend(loc="best")
        ax[1].grid(linestyle="--")    
        
        # Add final details
        ax[0].legend(loc="best")
        ax[0].grid(linestyle="--")
        ax[1].set_xlabel("Rice age (day)")
        ax[0].set_ylabel("Backscatter coefficient (dB)")
        ax[1].set_ylabel("NDWI")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{row.ext_act_id}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})")
        
        # Savefig
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210301\normal", f"{strip_id}_{count+1}.png"), bbox_inches="tight")
        count+=1
        if count == 21:
            break
    
    count = 0
    df_sample = df_flood.groupby(["ext_act_id"]).sample(n=1).head(50)
    for i, (index, row) in enumerate(df_sample.iterrows()):
        modis = df_ndwi_modis[df_ndwi_modis["row_col"] == df_mapping_modis.loc[df_mapping_modis["ext_act_id"] == row.ext_act_id, "row_col"].values[0]]
        modis = modis[modis["year"] == row.final_plant_date.year].squeeze()
        if len(modis) == 0:
            continue
        
        plt.close("all")
        fig, ax = initialize_plot(mean_normal, ci_normal, ylim=(-20, 0))
        
        # Plot temporal
        ax[0].plot(row[columns].values, linestyle="--", marker='o', color="blue", label="Sample(Flood)")
        
        # # Plot mean, median (age1)
        # ax[0].hlines(row["median(age1)"], xmin=0, xmax=6.5, linestyle="--", linewidth=2.5, color="orange", label="Median (Age1)")
        
        # # Plot mean, median (age2)
        # ax[0].hlines(row["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--", linewidth=2.5, color="gray", label="Median (age2)")
        
        # # Plot mean, median (age3)
        # ax[0].hlines(row["median(age3)"], xmin=15.0, xmax=29, linestyle="--", linewidth=2.5, color="purple", label="Median (age3)")
        
        # Draw start date
        ax[0].axvline(row.flood_column, color="red", linestyle="--")
        ax[0].text(row.flood_column, ax[0].get_ylim()[1]+0.8, "Reported flood date", horizontalalignment="center", color="red")
        
        # Add modis
        ax[1].plot([2*i-1 for i in range(1, 15)], modis[columns_modis].values, linestyle="--", marker='o', color="blue", label="Modis-ndwi")
        ax[1].legend(loc="best")
        ax[1].grid(linestyle="--")
        
        # Add final details
        ax[0].legend(loc="best")
        ax[0].grid(linestyle="--")
        ax[0].set_xlabel("Rice age (day)")
        ax[0].set_ylabel("Backscatter coefficient (dB)")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{row.ext_act_id}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})")
        
        # Savefig
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210301\flood", f"{strip_id}_{count+1}.png"), bbox_inches="tight")
        count+=1
        if count == 21:
            break
#%%