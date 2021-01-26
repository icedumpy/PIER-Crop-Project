import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe, load_mapping
from icedumpy.io_tools import save_model, load_model, save_h5, load_h5
#%% Define functions
def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]

    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df

def get_df_flood_nonflood(root_df_flood, root_df_nonflood, p, strip_id):
    df_flood = load_s1_flood_nonflood_dataframe(root_df_flood, p, strip_id=strip_id)
    df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p, strip_id=strip_id)

    # Drop row with nan (any)
    columns_pixel_values = df_flood.columns[df_flood.columns.str.contains(r'^t\d+$')].tolist()
    df_flood = df_flood.dropna(subset=columns_pixel_values)
    df_nonflood = df_nonflood.dropna(subset=columns_pixel_values)

    # Drop any zero or negative
    df_flood = df_flood.loc[~(df_flood[columns_pixel_values] <= 0).any(axis=1)]
    df_nonflood = df_nonflood.loc[~(df_nonflood[columns_pixel_values] <= 0).any(axis=1)]
    return df_flood, df_nonflood

def get_better_flood_date(arr_pixel, arr_diff, window_size, index_middle):
    list_index_flood = []
    for row in range(arr_pixel.shape[0]):
        argsort_pixel = np.argsort(arr_pixel[row, window_size-1:])+window_size-1
        argsort_diff = np.argsort(arr_diff[row, window_size-1:])+window_size-1
        
        index = None
        for pixel in argsort_pixel[:window_size]:
            if pixel in argsort_diff[:window_size]:
                index = pixel
                break
        if index is None:
            index = argsort_pixel[0]
        
        list_index_flood.append(index)
    return list_index_flood

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def histsubplot(df, group_name, column, ax, color="salmon", bins=20):
    sns.histplot(df.loc[df["Group"] == group_name, column],
                 kde=True, color=color, bins=bins, ax=ax)
    ax.set_title(group_name)
#%% Define directories
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_save_plot = r"F:\CROP-PIER\CROP-WORK\Presentation\20201214\Plot"
root_save_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1"
#%%  Define parameters
list_strip_id = ["302", "303", "304", "305", "306", "401", "402", "403"]
sat_type = "S1AB"
p = None
tier = [1]
all_touched = False
#%% Define parameters
strip_id = "402"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
root_df_nonflood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"

# Add strip_id to path
root_df_flood = os.path.join(root_df_flood, f"{sat_type.lower()}_flood_pixel")
root_df_nonflood = os.path.join(root_df_nonflood, f"{sat_type.lower()}_nonflood_pixel")
os.makedirs(os.path.join(root_save_plot, sat_type, strip_id, "raw_pixel"), exist_ok=True)
#%% Load dfflood, df_nonflood and drop nan
df_flood, df_nonflood = get_df_flood_nonflood(root_df_flood, root_df_nonflood, p, strip_id)
columns_pixel_values = df_flood.columns[df_flood.columns.str.match(r"^t\d+$")].tolist()

df_flood[columns_pixel_values] = 10*np.log10(df_flood[columns_pixel_values])
df_nonflood[columns_pixel_values] = 10*np.log10(df_nonflood[columns_pixel_values])
#%% Select a new proper flood date
window_size = 3
index_middle = len(columns_pixel_values)//2
list_index_flood = get_better_flood_date(df_flood[columns_pixel_values].values, 
                                         df_flood[columns_pixel_values].diff(axis=1).values, 
                                         window_size, index_middle)

arr = df_flood[columns_pixel_values].values
list_flood_values = []
for i in tqdm(range(len(df_flood))):
    list_flood_values.append(arr[i, list_index_flood[i]-2:list_index_flood[i]+1])
del arr

columns_pixel_values_new = [f"t{i}" for i in range(1, window_size+1)]
df_flood = df_flood.drop(columns=columns_pixel_values)
df_flood[columns_pixel_values_new] = list_flood_values

df_nonflood[columns_pixel_values_new] = df_nonflood[["t3", "t4", "t5"]]
df_nonflood = df_nonflood.drop(columns=list(set(columns_pixel_values)-set(columns_pixel_values_new)))
#%%
# =============================================================================
# Create train, test sample data
# =============================================================================
df_sample = pd.concat([df_flood, df_nonflood])
if sat_type == "S1A":
    df_sample = df_sample.loc[df_sample["ANCHOR_DATE"] >= datetime.datetime(2018, 6, 1)]

if (all_touched == False):
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id, list_p=list(map(str, df_sample["PLANT_PROVINCE_CODE"].unique())))
    df_mapping = df_mapping.loc[df_mapping["is_within"]]
    df_sample = pd.merge(df_sample, df_mapping, how='inner', on=["new_polygon_id", "row", "col"], left_index=True)

df_sample = df_sample.assign(label=(df_sample["loss_ratio"] != 0).astype("uint8"))
df_sample_drop_dup = df_sample.drop_duplicates(subset=columns_pixel_values_new)

if (all_touched == False) and (len(tier) != 2):
    df_sample_drop_dup = df_sample_drop_dup[df_sample_drop_dup["tier"].isin(tier)]
    
df_sample_drop_dup = df_sample_drop_dup[(df_sample_drop_dup["ext_act_id"].isin(np.random.choice(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 0, "ext_act_id"].unique(), len(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 1, "ext_act_id"].unique()), replace=False))) | (df_sample_drop_dup["label"] == 1)]
df_sample_drop_dup = df_sample_drop_dup.sample(frac=1.0)

# =============================================================================
# Create train-test samples
# =============================================================================
df_test = df_sample_drop_dup.loc[((df_sample_drop_dup["ext_act_id"]%10).isin([8, 9]))]
#%%
#%% This part is for loss_bins
model = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-modified\{strip_id}_RF_raw_pixel_values.joblib")
#%%
df_train = df_sample_drop_dup.loc[(~(df_sample_drop_dup["ext_act_id"]%10).isin([8, 9]))]
df_train = df_train.assign(predict_proba = model.predict_proba(df_train[columns_pixel_values_new])[:, 1])
df_train = df_train.groupby(["ext_act_id"]).agg({"predict_proba":["mean", "std"], "loss_ratio":"mean"})
df_train = df_train.fillna(0)
df_train.columns = df_train.columns.droplevel()
df_train.columns = ["mean_predict_proba", "std_predict_proba", "loss_ratio"]

df_train.loc[df_train["loss_ratio"] == 0, "bin"] = 0
df_train.loc[(df_train["loss_ratio"] > 0) & (df_train["loss_ratio"] < 0.8), "bin"] = 1
df_train.loc[(df_train["loss_ratio"] >= 0.8) & (df_train["loss_ratio"] <= 1.0), "bin"] = 2
#%%
sns.kdeplot(data=df_train, x="mean_predict_proba", hue="bin", palette=sns.color_palette("husl", 3))
#%%
threshold_01 = 0.1
threshold_12 = 0.2
df_train.loc[df_train["mean_predict_proba"] < threshold_01, "pred"] = 0
df_train.loc[(df_train["mean_predict_proba"] >= threshold_01) & (df_train["mean_predict_proba"] < threshold_12), "pred"] = 1
df_train.loc[df_train["mean_predict_proba"] >= threshold_12, "pred"] = 2
#%%
from icedumpy.eval_tools import kappa_from_cnf_matrix
for threshold_01 in np.arange(0.05, 1.00, 0.05):
    for threshold_12 in np.arange(threshold_01+0.05, 1.00, 0.05):
        df_train.loc[df_train["mean_predict_proba"] < threshold_01, "pred"] = 0
        df_train.loc[(df_train["mean_predict_proba"] >= threshold_01) & (df_train["mean_predict_proba"] < threshold_12), "pred"] = 1
        df_train.loc[df_train["mean_predict_proba"] >= threshold_12, "pred"] = 2
        
        plt.close('all')
        fig, ax = plt.subplots()
        (2*df_train["loss_ratio"] - df_train["pred"]).hist(bins=20, ax=ax)
        ax.set_title(f"{threshold_01:.2f}, {threshold_12:.2f}")
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210113\Fig", f"{threshold_01:.2f}, {threshold_12:.2f}.png"))
#%%
import plotly
import numpy as np
plotly.offline.init_notebook_mode()

total_days = 3

data = list()

for day in range(total_days):
    data.append(plotly.graph_objs.Histogram(
        x=np.random.randn(500) + day * 0.5,
        histnorm='count',
        name='Day {}, control'.format(day),
        visible=day < 1
      )
    )
    data.append(plotly.graph_objs.Histogram(
        x=np.random.randn(500) + day,
        histnorm='count',
        name='Day {}, experimental'.format(day),
        visible=day < 1
      )
    )

steps = list()
for i in range(total_days):
    step = dict(
        method='restyle',
        args=['visible', [False] * total_days * 2],
        label='Day {}'.format(i)
    )
    step['args'][1][i * 2] = True
    step['args'][1][i * 2 + 1] = True
    steps.append(step)

sliders = [dict(
    active=0,
    steps=steps
)]

layout = dict(sliders=sliders)
fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)







