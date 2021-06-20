import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from icedumpy.io_tools import load_model, load_h5
from icedumpy.df_tools import set_index_for_loc
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

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def plot_hist_each_class(df, x):
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.histplot(df, x=x, hue="year(label)", stat="density", kde=True, common_norm=False, ax=ax)
    ax.set_title(x)
    return fig, ax

def sampling_data(df):
    list_year = [2018, 2019, 2020]
    
    # Step1: Find min loss_to_noloss_ratio of each "PAT"
    list_loss_to_noloss_ratio = []
    for pat, df_grp in df.groupby("PAT"):
        loss_to_noloss_ratio = len(df_grp.loc[df_grp["label"] == 1, "ext_act_id"].unique())/len(df_grp.loc[df_grp["label"] == 0, "ext_act_id"].unique())
        if loss_to_noloss_ratio == 0:
            loss_to_noloss_ratio = 1
        list_loss_to_noloss_ratio.append(loss_to_noloss_ratio)    
    min_loss_to_noloss_ratio = min(list_loss_to_noloss_ratio)
    
    # Step2: Down-sampling no loss case following "Imbalance Data Sampling Strategy" from P'Tee+
    list_df_loss = []
    list_df_noloss = []
    for pat, df_grp in df.groupby("PAT"):
        loss_to_noloss_ratio = len(df_grp.loc[df_grp["label"] == 1, "ext_act_id"].unique())/len(df_grp.loc[df_grp["label"] == 0, "ext_act_id"].unique())
        for year in list_year:
            df_year_grp = df_grp[df_grp["final_crop_year"] == year]
            total_loss = len(df_year_grp.loc[df_year_grp["label"] == 1, "ext_act_id"].unique())
            total_noloss = len(df_year_grp.loc[df_year_grp["label"] == 0, "ext_act_id"].unique())
            
            # Sampling
            if total_loss == 0:
                if loss_to_noloss_ratio == 0:
                    n_sample_noloss = min_loss_to_noloss_ratio*total_noloss
                else:
                    n_sample_noloss = loss_to_noloss_ratio*total_noloss
            else:
                n_sample_noloss = min(total_loss, total_noloss)
            n_sample_noloss = np.ceil(n_sample_noloss).astype("uint")
            n_sample_noloss = min(total_noloss, n_sample_noloss)            
            
            list_df_loss.append(df_year_grp[df_year_grp["label"] == 1])
            list_df_noloss.append(df_year_grp[df_year_grp["ext_act_id"].isin(df_year_grp.loc[df["label"] == 0, "ext_act_id"].drop_duplicates().sample(n=n_sample_noloss))])
            
    # Return data
    list_df = list_df_loss + list_df_noloss
    df = pd.concat(list_df, ignore_index=True)
    return df

def assign_drop_rise(df):
    df = df.copy()
    # Get sharpest drop, rise lag1, 2, 3
    df = df.assign(**{
        "sharp_drop_lag1" : df[columns_age1[-1:]+columns_age2+columns_age3+columns_age4].diff(periods=1, axis=1).min(axis=1),
        "sharp_drop_lag2" : df[columns_age1[-2:]+columns_age2+columns_age3+columns_age4].diff(periods=2, axis=1).min(axis=1),
        "sharp_drop_lag3" : df[columns_age1[-3:]+columns_age2+columns_age3+columns_age4].diff(periods=3, axis=1).min(axis=1),
        "sharp_drop_lag1_column" : df[columns_age1[-1:]+columns_age2+columns_age3+columns_age4].diff(periods=1, axis=1).idxmin(axis=1),
        "sharp_drop_lag2_column" : df[columns_age1[-2:]+columns_age2+columns_age3+columns_age4].diff(periods=2, axis=1).idxmin(axis=1),
        "sharp_drop_lag3_column" : df[columns_age1[-3:]+columns_age2+columns_age3+columns_age4].diff(periods=3, axis=1).idxmin(axis=1)
    })
    
    # Sharpest_drop
    df = df.assign(**{
        "sharpest_drop" : df[["sharp_drop_lag1", "sharp_drop_lag2", "sharp_drop_lag3"]].min(axis=1),
    })
    
    # Sharpest_drop  column
    df = df.assign(**{
        "sharpest_drop_column" : df.apply(lambda val: val[val[["sharp_drop_lag1", "sharp_drop_lag2", "sharp_drop_lag3"]].astype(float).idxmin()+"_column"], axis=1),
    })
    
    # Sharpest_drop age
    df.loc[df["sharpest_drop_column"].isin(columns_age2), "sharpest_drop_age"] = 0
    df.loc[df["sharpest_drop_column"].isin(columns_age3), "sharpest_drop_age"] = 1
    df.loc[df["sharpest_drop_column"].isin(columns_age4), "sharpest_drop_age"] = 2
    df["sharpest_drop_age"] = df["sharpest_drop_age"].astype("uint8")
    
    # Find sharpest rise
    arr = np.zeros((len(df), 1), dtype="float32")
    for i, (_, row) in enumerate(df.iterrows()):
        row_columns = columns[columns.index(row["sharpest_drop_column"])+1:]
        row_data = row[row_columns]
        sharpest_rise = max(row_data.diff(periods=1).max(), row_data.diff(periods=2).max(), row_data.diff(periods=3).max())
        if np.isnan(sharpest_rise):
            row_columns = columns_age1[-1:]+columns_age2+columns_age3+columns_age4
            row_data = row[row_columns]
            sharpest_rise = max(row_data.diff(periods=1).max(), row_data.diff(periods=2).max(), row_data.diff(periods=3).max())
        arr[i] = sharpest_rise
    df = df.assign(sharpest_rise = arr)
    
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
# %%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season-v3"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"

# Model hyperparameters
# Number of trees in random forest
n_estimators = [100, 200, 500]
criterion = ["gini", "entropy"]
max_features = ["sqrt", "log2", 0.2, 0.3, 0.4]
max_depth = [2, 5, 10]
min_samples_split = [2, 5, 10]
min_samples_leaf = [2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion' : criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)

# Define columns' group name
columns = [f"t{i}" for i in range(0, 30)]
columns_large = [f"t{i}" for i in range(0, 35)]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179
model_parameters = [
    'sharpest_drop', 'sharpest_rise', 'sharpest_drop_age',
    'drop_t-2', 'drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3',
    'drop_t4', 'drop_t5'
]
# %%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    # strip_id = "304"
    print(strip_id)
    model = load_model(os.path.join(root_model, f"{strip_id}_old.joblib"))
    model.n_jobs = -1
    dict_roc_params = load_h5(os.path.join(root_model, f"{strip_id}_metrics_params_old.h5"))
    threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold_roc"], selected_fpr=0.2)
    
    # Load df rice code
    df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
    df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]
    #%%
    # Load data
    print("Loading data")
    df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    df = df.loc[(df["ext_act_id"]%10).isin([8, 9])]
    df = df[df["in_season_rice_f"] == 1]
    df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
    df["PAT"] = df['PLANT_PROVINCE_CODE'].astype(str).str.zfill(2) + df['PLANT_AMPHUR_CODE'].astype(str).str.zfill(2) + df['PLANT_TAMBON_CODE'].astype(str).str.zfill(2)
    
    # Convert power to db
    print("Converting to dB")
    df = convert_power_to_db(df, columns_large)
    
    # Assign sharp drop, rise
    print("Assigning sharp drop")
    df = assign_drop_rise(df)
    
    # Drop duplicates
    print("Dropping duplicates")
    df = df.drop_duplicates(subset=columns_large)
    
    # Merge photo sensitivity
    print("Merging photo sensitivity")
    df = pd.merge(df, df_rice_code, on="BREED_CODE", how="inner")
    
    # Normalize [drop_t-1, drop_t5] by subtracting drop_t-2
    df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']] = df[['drop_t-1', 'drop_t0', 'drop_t1', 'drop_t2', 'drop_t3', 'drop_t4', 'drop_t5']].subtract(df['drop_t-2'], axis=0)
    #%%
    # Predict and thresholding
    df = df.assign(predict_proba = model.predict_proba(df[model_parameters])[:, -1])
    df = df.assign(predict = (df["predict_proba"] >= threshold).astype("uint8"))
    #%%
    df_plot = df.groupby("ext_act_id")[["loss_ratio", "label", "predict"]].mean()
    df_plot = df_plot.assign(digitized_loss_ratio = np.digitize(df_plot["loss_ratio"], [0, 0.25, 0.5, 0.75], right=True))
    df_plot = df_plot.assign(digitized_predict = np.digitize(df_plot["predict"], [0, 0.25, 0.5, 0.75], right=True))
    #%%
    cnf_matrix = metrics.confusion_matrix(df_plot["digitized_loss_ratio"], df_plot["digitized_predict"], normalize="true")
    #%%
    plt.close("all")
    fig, ax = plt.subplots()
    ax = sns.heatmap(cnf_matrix, annot=True, fmt=".3f",
                     xticklabels=["bin0", "bin1", "bin2", "bin3", "bin4"],
                     yticklabels=["bin0", "bin1", "bin2", "bin3", "bin4"],
                     cmap="Blues", cbar=False, ax=ax)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210617\cnf_matrix", f"{strip_id}.png"), bbox_inches="tight")
#%%




df.groupby("ext_act_id").size()










