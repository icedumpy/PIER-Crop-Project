import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import seaborn as sns
import geopandas as gpd
from sklearn import metrics
import matplotlib.pyplot as plt
from icedumpy.df_tools import set_index_for_loc
from icedumpy.io_tools import load_model, load_h5
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

def plot_sample(df):
    if type(df) == pd.core.frame.DataFrame:
        row = df.sample(n=1).squeeze()
    else:
        row = df.copy()
    fig, ax = initialize_plot(ylim=(-20, -5))
    ax.plot(row[columns].values, linestyle="--", marker="o", color="blue")

    # Plot mean, median (age1)
    ax.hlines(row["median(age1)"], xmin=0, xmax=6.5, linestyle="--",
              linewidth=2.5, color="orange", label="Median (Age1)")

    # Plot mean, median (age2)
    ax.hlines(row["median(age2)"], xmin=6.5, xmax=15.0, linestyle="--",
              linewidth=2.5, color="gray", label="Median (age2)")

    # Plot mean, median (age3)
    ax.hlines(row["median(age3)"], xmin=15.0, xmax=20, linestyle="--",
              linewidth=2.5, color="purple", label="Median (age3)")

    # Plot mean, median (age4)
    ax.hlines(row["median(age4)"], xmin=20.0, xmax=29, linestyle="--",
              linewidth=2.5, color="yellow", label="Median (age4)")
    try:
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{int(row.ext_act_id)}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})\n Loss ratio:{row.loss_ratio:.2f}")
    except:
        pass
    plt.grid(linestyle="--")
    return fig, ax

def plot_ext_act_id(df, ext_act_id):
    plt.close("all")
    plot_sample(df.loc[df["ext_act_id"] == ext_act_id])  

def assign_sharp_drop(df):
    df = df.copy()
    # Sharpest drop of each age
    df = df.assign(**{
       "sharp_drop": df[columns].diff(axis=1).dropna(axis=1).min(axis=1),
       "sharp_drop(age1)": df[columns_age1].diff(axis=1).dropna(axis=1).min(axis=1),
       "sharp_drop(age2)": df[columns_age1[-1:]+columns_age2].diff(axis=1).dropna(axis=1).min(axis=1),
       "sharp_drop(age3)": df[columns_age2[-1:]+columns_age3].diff(axis=1).dropna(axis=1).min(axis=1),
       "sharp_drop(age4)": df[columns_age3[-1:]+columns_age4].diff(axis=1).dropna(axis=1).min(axis=1),
    })
    
    # Get sharpest drop period (age1 or age2 or age3 or age4)
    df = df.assign(**{
        "sharpest_drop_period" : df[["sharp_drop(age2)", "sharp_drop(age3)", "sharp_drop(age4)"]].idxmin(axis=1)
    })
    # Get sharpest drop value and which "t{x}"
    df = df.assign(**{
        "sharpest_drop":df[["sharp_drop(age2)", "sharp_drop(age3)", "sharp_drop(age4)"]].min(axis=1),
        "sharpest_drop_column" : df[columns_age2+columns_age3+columns_age4].diff(axis=1).dropna(axis=1).idxmin(axis=1)
    })
    df["sharpest_drop_period"] = df["sharpest_drop_period"].str.slice(-2,-1).astype(int)
    
    # Get backscatter coeff before and after sharpest drop (-2, -1, 0, 1, 2, 3, 4, 5) before 12 days and after 30 days
    arr = np.zeros((len(df), 8), dtype="float32")
    for i, (_, row) in enumerate(df.iterrows()):
        column = int(row["sharpest_drop_column"][1:])
        arr[i] = row[[f"t{i}" for i in range(column-2, column+6)]].values
    
    df = df.assign(**{
        "drop_t-2" : arr[:, 0],
        "drop_t-1" : arr[:, 1],
        "drop_t0" :  arr[:, 2],
        "drop_t1" :  arr[:, 3],
        "drop_t2" :  arr[:, 4],
        "drop_t3" :  arr[:, 5],
        "drop_t4" :  arr[:, 6],
        "drop_t5" :  arr[:, 7],
    })
    return df

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_2020"
root_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season-v2"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210519"
root_shp = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged_shp"
root_shp_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202_shp"

path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
strip_id = "304"

# Define columns' group name
columns = [f"t{i}" for i in range(0, 30)]
columns_large = [f"t{i}" for i in range(0, 35)]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179

model = load_model(os.path.join(root_model, f"{strip_id}.joblib"))
model.n_jobs = -1
dict_roc_params = load_h5(os.path.join(root_model, f"{strip_id}_metrics_params.h5"))
threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold_roc"], selected_fpr=0.2)
model_parameters = ["sharpest_drop_period", "sharpest_drop", 'drop_t-2', 'drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']
#%%
# Load df pixel value (2018-2019)
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df.rename(columns = {"PLANT_PROVINCE_CODE" : "p_code"})
df = df[(df["ext_act_id"]%10).isin([8, 9])]
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] > 0, "ext_act_id"].unique())))) | (df["loss_ratio"] > 0)]

# Load df pixel value (2020)
df_2020 = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal_2020, file)) for file in os.listdir(root_df_s1_temporal_2020) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df_2020 = df_2020[df_2020[columns_large].isna().sum(axis=1) <= 3]
df_2020 = df_2020[(df_2020["ext_act_id"]%10).isin([8, 9])]
df_2020 = df_2020[(df_2020["ext_act_id"].isin(np.random.choice(df_2020.loc[df_2020["loss_ratio"] == 0, "ext_act_id"].unique(), len(df_2020.loc[df_2020["loss_ratio"] > 0, "ext_act_id"].unique())))) | (df_2020["loss_ratio"] > 0)]

# Convert power to db
print("Convert to dB")
df = convert_power_to_db(df, columns_large)
df_2020 = convert_power_to_db(df_2020, columns_large)

# Drop duplicates
print("Drop duplicates")
df = df.drop_duplicates(subset=columns_large)
df_2020 = df_2020.drop_duplicates(subset=columns_large)

# Assign shapr drop
print("Assign sharpdrop 2018, 2019")
df = assign_sharp_drop(df)
print("Assign sharpdrop 2020")
df_2020 = assign_sharp_drop(df_2020)

# Merge 2018, 2019, 2020
print("Merge df")
df_2020 = df_2020.drop(columns = ['is_within', 'tier', 'photo_sensitive_f'])
df = df[df_2020.columns]
df = pd.concat([df, df_2020])
df["year"] = df["final_plant_date"].dt.year
df["year(label)"] = df["year"].astype(str) + "(" + df["label"].astype(str) + ")"
df["p_code"] = df["p_code"].astype(int)
df["sharpest_drop_column"] = df["sharpest_drop_column"].str.slice(1).astype(int)


del df_2020

# Normalize [drop_t-1, drop_t5] by subtracting drop_t-2
df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']] = df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']].subtract(df['drop_t-2'], axis=0)
#%%
# Predict and thresholding
df = df.assign(predict_proba = model.predict_proba(df[model_parameters])[:, -1])
df = df.assign(predict = (df["predict_proba"] >= threshold).astype("uint8"))

# Pixel-level performance
cnf_matrix = metrics.confusion_matrix(df["label"], df["predict"])
dt = 100*cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])
fa = 100*cnf_matrix[0, 1]/(cnf_matrix[0, 0]+cnf_matrix[0, 1])
print(f"Out of sample pixel-level performance: dt({dt:.2f}%), fa({fa:.2f}%)")
#%%
## Plot-level performance
df_plot = df.groupby("ext_act_id").mean()[["sharpest_drop_column", "p_code", "year", "loss_ratio", "predict", "label"]]
df_plot.loc[df_plot["year"].isin([2018, 2019]), "shp_year"] = "2018-2019"
df_plot.loc[df_plot["year"].isin([2020]), "shp_year"] = "2020"
df_plot = df_plot.reset_index()
df_plot = df_plot.assign(**{"actual-predict": df_plot["loss_ratio"]-df_plot["predict"]})
#%%
# Plot error Flood
plt.close("all")
for disaster_type in ["Flood", "nonFlood"]:
    for year in df_plot["year"].unique().astype("int"):
        if disaster_type == "Flood":
            label = 1
        elif disaster_type == "nonFlood":
            label = 0
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.histplot(df_plot.loc[(df_plot["year"] == year) & (df_plot["label"] == label), "actual-predict"], 
                     stat="probability", ax=ax, bins="auto", kde=True)
        ax.set_xlabel("actual-predict")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(-1, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_title(f"{disaster_type}\nStrip_id: {strip_id}, Year: {year}")
        ax.grid("--")
        fig.savefig(os.path.join(root_save, "Fig", f"{disaster_type}_{strip_id}_{year}.png"), bbox_inches="tight")
# Plot error all in 1
plt.close("all")
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 9))
ax = ax.T.reshape(-1).tolist()
ax_iter = iter(ax)
for disaster_type in ["nonFlood", "Flood"]:
    for year in df_plot["year"].unique().astype("int"):
        if disaster_type == "Flood":
            label = 1
        elif disaster_type == "nonFlood":
            label = 0
        ax_curr = next(ax_iter)
        sns.histplot(df_plot.loc[(df_plot["year"] == year) & (df_plot["label"] == label), "actual-predict"], 
                     stat="probability", bins="auto", kde=True, label=f"{disaster_type}({year})", ax=ax_curr)
        ax_curr.set_xlim(-1, 1)
        ax_curr.set_ylim(0, 1)
        ax_curr.set_xticks(np.arange(-1, 1.1, 0.25))
        ax_curr.set_yticks(np.arange(0, 1.1, 0.25))
        # ax_curr.set_title(f"{disaster_type}\nStrip_id: {strip_id}, Year: {year}")
        ax_curr.grid("--")
        ax_curr.legend(loc=1)
fig.suptitle(f"Strip_id: {strip_id}")
fig.savefig(os.path.join(root_save, "Fig", f"{strip_id}.png"), bbox_inches="tight")
#%%
for (p_code, shp_year), df_plot_grp in df_plot.groupby(["p_code", "shp_year"]):
    if shp_year == "2018-2019":
        path_shp = os.path.join(root_shp, f"vew_polygon_id_plant_date_disaster_merged_p{int(p_code)}.shp")
    elif shp_year == "2020":
        path_shp = os.path.join(root_shp_2020, f"vew_polygon_id_plant_date_disaster_20210202_{int(p_code)}.shp")
    print(p_code, shp_year, path_shp)
    gdf = gpd.read_file(path_shp)
    gdf = gdf[gdf["ext_act_id"].isin(df_plot_grp["ext_act_id"])]
    gdf = pd.merge(gdf, df_plot_grp, on="ext_act_id", how="inner")
    gdf.to_file(os.path.join(root_save, "shp", f"{strip_id}_{int(p_code)}_{shp_year}.shp"))
#%%
ext_act_id = 9199554199
try:
    plot_ext_act_id(df, ext_act_id)
except:
    pass



