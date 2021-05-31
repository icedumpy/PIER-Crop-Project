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

def initialize_plot2(ylim1=(-20, 0), ylim2=(-10, 5)):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 9), nrows=1, ncols=2)

    # Draw group age
    ax[0].axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax[0].axvspan(6.5, 15, alpha=0.2, color='green')
    ax[0].axvspan(15.0, 20, alpha=0.2, color='yellow')
    ax[0].axvspan(20.0, 34, alpha=0.2, color='purple')

    [ax[0].axvline(i, color="black") for i in [6.5, 15, 20]]

    # Add group descriptions
    ax[0].text(3, ylim1[-1]-0.25, "0-40 days", horizontalalignment="center")
    ax[0].text(10.5, ylim1[-1]-0.25, "40-90 days", horizontalalignment="center")
    ax[0].text(17.5, ylim1[-1]-0.25, "90-120 days", horizontalalignment="center")
    ax[0].text(27.5, ylim1[-1]-0.25, "120+ days", horizontalalignment="center")

    # Set y limits
    ax[0].set_ylim(ylim1)
    ax[1].set_ylim(ylim2)
    
    # Add more ticks
    ax[0].set_xticks(range(0, 35, 2))
    ax[0].set_yticks(np.arange(*ylim1))
    ax[1].set_yticks(np.arange(*ylim2))
    
    return fig, ax

def plot_sample(df):
    if type(df) == pd.core.frame.DataFrame:
        row = df.sample(n=1).squeeze()
    else:
        row = df.copy()
    fig, ax = initialize_plot(ylim=(-20, -5))
    ax.plot(row[columns].values, linestyle="--", marker="o", color="blue")

    try:
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
    except:
        pass
    
    try:
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{int(row.ext_act_id)}\nPolygon area:{row.polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})\n Loss ratio:{row.loss_ratio:.2f}")
    except:
        pass
    ax.grid(linestyle="--")
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
root_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season-v2"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_malison_2020"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_malison_2020_predicted"

# Define columns' group name
columns = [f"t{i}" for i in range(0, 30)]
columns_large = [f"t{i}" for i in range(0, 35)]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179
model_parameters = ["sharpest_drop_period", "sharpest_drop", 'drop_t-2', 'drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']
#%%
list_df = []
for strip_id in np.unique([file.split(".")[0][-3:] for file in os.listdir(root_df_s1_temporal)]):
    print(strip_id)
    # Load model
    model = load_model(os.path.join(root_model, f"{strip_id}.joblib"))
    model.n_jobs = -1
    dict_roc_params = load_h5(os.path.join(root_model, f"{strip_id}_metrics_params.h5"))
    threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold_roc"], selected_fpr=0.2)

    # Load df
    df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    df[columns_large] = df[columns_large].replace({0: np.nan})
    df = df[df[columns_large].isna().sum(axis=1) <= 4]

    # Convert power to db
    df = convert_power_to_db(df, columns_large)
    
    # Assign sharp drop
    df = assign_sharp_drop(df)

    # Normalize
    df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']] = df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']].subtract(df['drop_t-2'], axis=0)

    # Predict and thresholding
    df = df.assign(predict_proba = model.predict_proba(df[model_parameters])[:, -1])
    df = df.assign(predict = (df["predict_proba"] >= threshold).astype("uint8"))
    df_plot = df.groupby("warranty_i").agg({"PLANT_PROV":pd.Series.mode,
                                            "predict":np.mean})
    df_plot = df_plot.rename(columns={"predict":"predicted_loss_ratio"})
    df_plot = df_plot.reset_index(drop=False)
    df_plot.to_parquet(os.path.join(root_save, f"{strip_id}.parquet"))
    list_df.append(df_plot)
#%%
df_plot = pd.concat(list_df, ignore_index=True)
df_plot = df_plot.groupby("warranty_i").agg({"PLANT_PROV":pd.Series.mode,
                                             "predicted_loss_ratio":np.mean})
df_plot = df_plot.reset_index(drop=False)
df_plot.to_parquet(os.path.join(root_save, "Malison-result.parquet"))
