import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from numba import jit
import geopandas as gpd
from sklearn import metrics
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_model, load_h5
from icedumpy.plot_tools import plot_roc_curve, plot_precision_recall_curve, set_roc_plot_template
# %%
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

@jit(nopython=True)
def get_area_under_median(arr_min_index, arr_value_under_median):
    arr_area_under_median = np.zeros_like(arr_min_index, dtype=np.float32)
    # 0:Consecutive first, 1:No consecutive, 2:Consecutive last
    arr_is_consecutive = np.ones_like(arr_min_index, dtype=np.uint8)
    arr_consecutive_size = np.zeros_like(arr_min_index, dtype=np.uint8)
    for i, (index_min, value_under_median) in enumerate(zip(arr_min_index, arr_value_under_median)):
        arr_group = np.zeros((arr_value_under_median.shape[-1]))
        group = 1
        for j, value in enumerate(value_under_median):
            if value > 0:
                arr_group[j] = group
            else:
                group += 1

        arr_where_min_group = np.where(arr_group == arr_group[index_min])[0]
        area_under_median = value_under_median[arr_where_min_group].sum()
        arr_area_under_median[i] = area_under_median
        if arr_where_min_group[0] == 0:
            arr_is_consecutive[i] = 0
        elif arr_where_min_group[-1] == arr_value_under_median.shape[-1]-1:
            arr_is_consecutive[i] = 2
        arr_consecutive_size[i] = len(arr_where_min_group)
    return arr_area_under_median, arr_is_consecutive, arr_consecutive_size

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

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
    ax.text(24.5, ylim[-1]+0.25, "120+ days", horizontalalignment="center")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(30))
    ax.set_yticks(np.arange(*ylim))

    return fig, ax

def plot_sample(df):
    if type(df) == pd.core.frame.DataFrame:
        row = df.sample(n=1).squeeze()
    else:
        row = df.copy()
    fig, ax = initialize_plot(ylim=(-20, -5))
    ax.plot(df.loc[df["ext_act_id"] == ext_act_id, columns].T.values, linestyle="--", marker="o")
    try:
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{int(row.ext_act_id)}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})\n Loss ratio:{row.loss_ratio:.2f}")
    except:
        pass
    plt.grid(linestyle="--")
    return fig, ax

def plot_ext_act_id(df, ext_act_id):
    plt.close("all")
    print(df.loc[df["ext_act_id"] == ext_act_id, "predict_proba"])
    plot_sample(df.loc[df["ext_act_id"] == ext_act_id])    

#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_2020"
root_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season"
root_shp = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202_shp"
strip_id = "304"
#%%
model = load_model(os.path.join(root_model, f"{strip_id}.joblib"))
dict_roc_params = load_h5(os.path.join(root_model, f"{strip_id}_metrics_params.h5"))

# Define columns' group name
columns = [f"t{i}" for i in range(0, 30)]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179
#%%
# Load df pixel value
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df[columns].isna().sum(axis=1) <= 3]
df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1.0)]

# Convert power to db
df = convert_power_to_db(df, columns)

# Drop duplicates
df = df.drop_duplicates(subset=columns)
#%%
# Assign median
df = df.assign(**{"median": df[columns].median(axis=1),
                  "median(age1)": df[columns_age1].median(axis=1),
                  "median(age2)": df[columns_age2].median(axis=1),
                  "median(age3)": df[columns_age3].median(axis=1),
                  "median(age4)": df[columns_age4].median(axis=1),
                  })

# Assign median - min
df = df.assign(**{"median-min": df["median"]-df[columns].min(axis=1),
                  "median-min(age1)": df["median(age1)"]-df[columns_age1].min(axis=1),
                  "median-min(age2)": df["median(age2)"]-df[columns_age2].min(axis=1),
                  "median-min(age3)": df["median(age3)"]-df[columns_age3].min(axis=1),
                  "median-min(age4)": df["median(age4)"]-df[columns_age4].min(axis=1),
                  })

# Assign min value
df = df.assign(**{"min": df[columns].min(axis=1),
                  "min(age1)": df[columns_age1].min(axis=1),
                  "min(age2)": df[columns_age2].min(axis=1),
                  "min(age3)": df[columns_age3].min(axis=1),
                  "min(age4)": df[columns_age4].min(axis=1),
                  })

# Assign max-min value
df = df.assign(**{"max-min": df[columns].max(axis=1)-df[columns].min(axis=1),
                  "max-min(age1)": df[columns_age1].max(axis=1)-df[columns_age1].min(axis=1),
                  "max-min(age2)": df[columns_age2].max(axis=1)-df[columns_age2].min(axis=1),
                  "max-min(age3)": df[columns_age3].max(axis=1)-df[columns_age3].min(axis=1),
                  "max-min(age4)": df[columns_age4].max(axis=1)-df[columns_age4].min(axis=1),
                  })

# =============================================================================
# Assign area under median, consecutive under median
# =============================================================================
arr_area_under_median, _, arr_consecutive_size = get_area_under_median(np.argmax(df[columns].eq(df[columns].min(
    axis=1), axis=0).values, axis=1), (-df[columns].sub(df["median"], axis=0)).clip(lower=0, upper=None).values)
arr_area_under_median_age1, arr_is_consecutive_age1, arr_consecutive_size_age1 = get_area_under_median(np.argmax(df[columns_age1].eq(
    df[columns_age1].min(axis=1), axis=0).values, axis=1), (-df[columns_age1].sub(df["median(age1)"], axis=0)).clip(lower=0, upper=None).values)
arr_area_under_median_age2, arr_is_consecutive_age2, arr_consecutive_size_age2 = get_area_under_median(np.argmax(df[columns_age2].eq(
    df[columns_age2].min(axis=1), axis=0).values, axis=1), (-df[columns_age2].sub(df["median(age2)"], axis=0)).clip(lower=0, upper=None).values)
arr_area_under_median_age3, arr_is_consecutive_age3, arr_consecutive_size_age3 = get_area_under_median(np.argmax(df[columns_age3].eq(
    df[columns_age3].min(axis=1), axis=0).values, axis=1), (-df[columns_age3].sub(df["median(age3)"], axis=0)).clip(lower=0, upper=None).values)
arr_area_under_median_age4, arr_is_consecutive_age4, arr_consecutive_size_age4 = get_area_under_median(np.argmax(df[columns_age4].eq(
    df[columns_age4].min(axis=1), axis=0).values, axis=1), (-df[columns_age4].sub(df["median(age4)"], axis=0)).clip(lower=0, upper=None).values)

# 1 <-> 2, 2 <-> 1
arr_index = np.where((arr_is_consecutive_age1 == 2) &
                     (arr_is_consecutive_age2 == 0))[0]
arr_consecutive_size_age1[arr_index] += arr_consecutive_size_age2[arr_index]
arr_consecutive_size_age2[arr_index] = arr_consecutive_size_age1[arr_index]
arr_area_under_median_age1[arr_index] += arr_area_under_median_age2[arr_index]
arr_area_under_median_age2[arr_index] = arr_area_under_median_age1[arr_index]

# 2 <-> 3, 3 <-> 2
arr_index = np.where((arr_is_consecutive_age2 == 2) &
                     (arr_is_consecutive_age3 == 0))[0]
arr_consecutive_size_age2[arr_index] += arr_consecutive_size_age3[arr_index]
arr_consecutive_size_age3[arr_index] = arr_consecutive_size_age2[arr_index]
arr_area_under_median_age2[arr_index] += arr_area_under_median_age3[arr_index]
arr_area_under_median_age3[arr_index] = arr_area_under_median_age2[arr_index]

# 3 <-> 4, 4 <-> 3
arr_index = np.where((arr_is_consecutive_age3 == 2) &
                     (arr_is_consecutive_age4 == 0))[0]
arr_consecutive_size_age3[arr_index] += arr_consecutive_size_age4[arr_index]
arr_consecutive_size_age4[arr_index] = arr_consecutive_size_age3[arr_index]
arr_area_under_median_age3[arr_index] += arr_area_under_median_age4[arr_index]
arr_area_under_median_age4[arr_index] = arr_area_under_median_age3[arr_index]

# Assign sum of cunsecutive values under median (around min value)
df = df.assign(**{"area-under-median": arr_area_under_median,
                  "area-under-median(age1)": arr_area_under_median_age1,
                  "area-under-median(age2)": arr_area_under_median_age2,
                  "area-under-median(age3)": arr_area_under_median_age3,
                  "area-under-median(age4)": arr_area_under_median_age4
                  })

# Assign count of cunsecutive values under median (around min value)
df = df.assign(**{"count-under-median": arr_consecutive_size,
                  "count-under-median(age1)": arr_consecutive_size_age1,
                  "count-under-median(age2)": arr_consecutive_size_age2,
                  "count-under-median(age3)": arr_consecutive_size_age3,
                  "count-under-median(age4)": arr_consecutive_size_age4
                  })

# Conclude some of the parameters
df = df.assign(**{"median(min)": df[["median(age1)", "median(age2)", "median(age3)", "median(age4)"]].min(axis=1),
                  "median-min(max)": df[["median-min(age1)", "median-min(age2)", "median-min(age3)", "median-min(age4)"]].max(axis=1),
                  "max-min(max)": df[["max-min(age1)", "max-min(age2)", "max-min(age3)", "max-min(age4)"]].max(axis=1),
                  "area-under-median(max)": df[["area-under-median(age1)", "area-under-median(age2)", "area-under-median(age3)", "area-under-median(age4)"]].max(axis=1),
                  "count-under-median(max)": df[["count-under-median(age1)", "count-under-median(age2)", "count-under-median(age3)", "count-under-median(age4)"]].max(axis=1)
                  })
#%%
model_parameters = ['median', 'median(age1)', 'median(age2)', 'median(age3)', 'median(age4)',
                    'median-min', 'median-min(age1)', 'median-min(age2)', 'median-min(age3)', 'median-min(age4)',
                    'min', 'min(age1)', 'min(age2)', 'min(age3)', 'min(age4)',
                    'max-min', 'max-min(age1)', 'max-min(age2)', 'max-min(age3)', 'max-min(age4)',
                    'area-under-median', 'area-under-median(age1)', 'area-under-median(age2)',
                    'area-under-median(age3)', 'area-under-median(age4)',
                    'count-under-median', 'count-under-median(age1)', 'count-under-median(age2)',
                    'count-under-median(age3)', 'count-under-median(age4)',
                    'photo_sensitive_f']
#%%
df = df.assign(predict_proba = model.predict_proba(df[model_parameters].values)[:, 1])
threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold_roc"], selected_fpr=0.1)
print(threshold)
df["predict"] = (df["predict_proba"] >= threshold).astype("uint8")
#%%
ice = df.groupby(["ext_act_id"]).mean()[["loss_ratio", "predict", "label"]]
ice["Loss ratio - Predicted loss ratio"] = ice["loss_ratio"] - ice["predict"]
#%%
df_temp = df[df["loss_ratio"] >= 0.8].copy()
df_temp["p_code"] = df_temp["p_code"].astype("uint8")
df_temp = df_temp.groupby(["ext_act_id"]).mean()[["p_code", "loss_ratio", "predict"]]
df_temp = df_temp.reset_index()
#%%
for p, df_temp_grp in df_temp.groupby(["p_code"]):
    path_shp = os.path.join(root_shp, f"vew_polygon_id_plant_date_disaster_20210202_{int(p)}.shp")
    gdf = gpd.read_file(path_shp)
    gdf = gdf[gdf["ext_act_id"].isin(df_temp_grp["ext_act_id"])]
    gdf = pd.merge(gdf, df_temp_grp, on="ext_act_id", how="inner")
    gdf.to_file(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210402\shp", f"{strip_id}_{int(p)}.shp"))
#%%
ext_act_id = 9233734134
plot_ext_act_id(df, ext_act_id)

















