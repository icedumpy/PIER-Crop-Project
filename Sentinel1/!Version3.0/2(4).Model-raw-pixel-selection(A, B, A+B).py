import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe, load_mapping
from icedumpy.io_tools import save_model, save_h5
from icedumpy.eval_tools import kappa_from_cnf_matrix
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
    
def create_polygon_level_train_test_df(df_sample_drop_dup, model, columns_pixel_values, train_or_test):
    if train_or_test == "train":
        df = df_sample_drop_dup.loc[(~(df_sample_drop_dup["ext_act_id"]%10).isin([8, 9]))]
    elif train_or_test == "test":
        df = df_sample_drop_dup.loc[((df_sample_drop_dup["ext_act_id"]%10).isin([8, 9]))]
        
    df = df.assign(predict_proba = model.predict_proba(df[columns_pixel_values])[:, 1])
    df = df.groupby(["ext_act_id"]).agg({"predict_proba":["mean", "std"], "loss_ratio":"mean"})
    # df = df.dropna()
    df = df.fillna(0)
    df.columns = df.columns.droplevel()
    df.columns = ["mean_predict_proba", "std_predict_proba", "loss_ratio"]
    
    df.loc[df["loss_ratio"] == 0.00, "bin"] = 0
    df.loc[(df["loss_ratio"] > 0.00) & (df["loss_ratio"] <= 0.25), "bin"] = 1
    df.loc[(df["loss_ratio"] > 0.25) & (df["loss_ratio"] <= 0.50), "bin"] = 2
    df.loc[(df["loss_ratio"] > 0.50) & (df["loss_ratio"] <= 0.75), "bin"] = 3
    df.loc[(df["loss_ratio"] > 0.75) & (df["loss_ratio"] <= 1.00), "bin"] = 4 
    return df
#%% Define directories
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_save_plot = r"F:\CROP-PIER\CROP-WORK\Presentation\20201214\Plot"
root_save_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB"
#%%  Define parameters
list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
sat_type = "S1AB"
p = None
tier = [1]
all_touched = False
#%% Define parameters
# for strip_id in list_strip_id:
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

# df_flood[columns_pixel_values] = 10*np.log10(df_flood[columns_pixel_values])
# df_nonflood[columns_pixel_values] = 10*np.log10(df_nonflood[columns_pixel_values])
#%% Select a new proper flood date : Route 1
window_size = 3
index_middle = len(columns_pixel_values)//2
list_index_flood = get_better_flood_date(df_flood[columns_pixel_values].values, 
                                         df_flood[columns_pixel_values].diff(axis=1).values, 
                                         window_size, index_middle)

arr = df_flood[columns_pixel_values].values
age = (df_flood["ANCHOR_DATE"] - df_flood["final_plant_date"]).dt.days.values
age = age-6*(np.array(list_index_flood)-4)
df_flood["ANCHOR_DATE"] = df_flood["ANCHOR_DATE"]+np.array([datetime.timedelta(int(item)) for item in 6*(np.array(list_index_flood)-4)])
# Digitize 1
# [0, 40] = 1
# (40, 90] = 2
# [90, inf ) = 3
bins = [0, 40, 90]

age = np.digitize(age, bins=bins, right=True)
list_flood_values = []
for i in tqdm(range(len(df_flood))):
    list_flood_values.append(np.hstack([arr[i, list_index_flood[i]-2:list_index_flood[i]+1], age[i]]))
del arr

columns_pixel_values_new = [f"t{i}" for i in range(1, window_size+1)] + ["age"]
df_flood = df_flood.drop(columns=columns_pixel_values)
df_flood[columns_pixel_values_new] = list_flood_values

# Digitize 2
# age = (df_nonflood["ANCHOR_DATE"] - df_nonflood["final_plant_date"]).dt.days.values.reshape(-1, 1)
age = np.digitize((df_nonflood["ANCHOR_DATE"] - df_nonflood["final_plant_date"]).dt.days.values.reshape(-1, 1), bins=bins, right=True)
df_nonflood[columns_pixel_values_new] = np.hstack([df_nonflood[["t3", "t4", "t5"]].values, age])
df_nonflood = df_nonflood.drop(columns=list(set(columns_pixel_values)-set(columns_pixel_values_new)))

#%% Route 2
# bins = [0, 40, 90]
# columns_pixel_values_new = ["t3", "t4", "t5"]

# df_flood = df_flood.drop(columns=["t1", "t2", "t6", "t7", "t8"])
# df_nonflood = df_nonflood.drop(columns=["t1", "t2", "t6", "t7", "t8"])

# df_flood = df_flood.assign(age = np.digitize((df_flood["ANCHOR_DATE"] - df_flood["final_plant_date"]).dt.days, bins=bins))
# df_nonflood = df_nonflood.assign(age = np.digitize((df_nonflood["ANCHOR_DATE"] - df_nonflood["final_plant_date"]).dt.days, bins=bins))
#%%
# =============================================================================
# Create train, test sample data
# =============================================================================
df_sample = pd.concat([df_flood, df_nonflood])
df_sample = df_sample.loc[df_sample["age"] > 0]

# Add month
# df_sample = df_sample.assign(month=df_sample["ANCHOR_DATE"].dt.month)
# columns_pixel_values_new = columns_pixel_values_new + ["month"]

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

# df_sample_drop_dup = df_sample_drop_dup.loc[(df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"].between(0.8, 1.0))]
df_sample_drop_dup = df_sample_drop_dup[(df_sample_drop_dup["ext_act_id"].isin(np.random.choice(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 0, "ext_act_id"].unique(), len(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 1, "ext_act_id"].unique()), replace=False))) | (df_sample_drop_dup["label"] == 1)]
df_sample_drop_dup = df_sample_drop_dup.sample(frac=1.0)
#%%
# =============================================================================
# Create train-test samples (for pixel-level model (1))
# =============================================================================
df_train = df_sample_drop_dup.loc[(~(df_sample_drop_dup["ext_act_id"]%10).isin([8, 9])) & ((df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"] >= 0.8))]
df_test = df_sample_drop_dup.loc[((df_sample_drop_dup["ext_act_id"]%10).isin([8, 9])) & ((df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"] >= 0.8))]

# if (len(df_train) == 0) or (len(df_test) == 0):
#     continue
# =============================================================================
# Fit model (pixel-level model) ♣
# =============================================================================
model1 = RandomForestClassifier(min_samples_leaf=5, max_depth=10, min_samples_split=10,
                               verbose=0, n_jobs=-1, random_state=42)
model1.fit(df_train[columns_pixel_values_new].values, df_train["label"].values)

plt.close('all')
fig, ax = plt.subplots(figsize=(16, 9))
ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model1, df_train[columns_pixel_values_new].values, df_train["label"].values, "train", color="g", ax=ax)
ax, _, _, _, _, _ = plot_roc_curve(model1, df_test[columns_pixel_values_new].values, df_test["label"].values, "test", color='b', ax=ax)
ax = set_roc_plot_template(ax)
ax.set_title(f'ROC Curve: {strip_id}\nAll_touched({all_touched}), Tier{tuple(tier)}\nTrain samples: Flood:{(df_train["label"] == 1).sum():,}, Non-Flood:{(df_train["label"] == 0).sum():,}\nTest samples: Flood:{(df_test["label"] == 1).sum():,}, Non-Flood:{(df_test["label"] == 0).sum():,}')
fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210121\Fig\ROC", f"{strip_id}_ROC_at({all_touched})_tier{tuple(tier)}.png"))
plt.close('all')
#%%
# =============================================================================
# Create train-test samples (for polygon-based model (2))
# =============================================================================
df_train = create_polygon_level_train_test_df(df_sample_drop_dup, model1, columns_pixel_values=columns_pixel_values_new, train_or_test="train")
df_test  = create_polygon_level_train_test_df(df_sample_drop_dup, model1, columns_pixel_values=columns_pixel_values_new, train_or_test="test")
#%% Classifier
x_train = df_train[["mean_predict_proba", "std_predict_proba"]].values
y_train = df_train[["bin"]].values.reshape(-1)

x_test = df_test[["mean_predict_proba", "std_predict_proba"]].values
y_test = df_test[["bin"]].values.reshape(-1)
#%%
model2 = RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=5, n_estimators=25)
model2.fit(x_train, y_train)

plt.close('all')
y_pred = model2.predict(x_test)
fig, ax = plt.subplots(figsize=(16, 9))
metrics.plot_confusion_matrix(model2, x_test, y_test, normalize="true", colorbar=False, ax=ax)
ax.set_title(f"{strip_id}_test")
fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210121\Fig\cnf_matrix", f"{strip_id}_test"))

plt.close('all')
y_pred = model2.predict(x_train)
fig, ax = plt.subplots(figsize=(16, 9))
metrics.plot_confusion_matrix(model2, x_train, y_train, normalize="true", colorbar=False, ax=ax)
ax.set_title(f"{strip_id}_train")
fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210121\Fig\cnf_matrix", f"{strip_id}_train"))
#%%
path_model1 = os.path.join(root_save_model, f"{strip_id}-w-age", "model1.joblib")
path_model2 = os.path.join(root_save_model, f"{strip_id}-w-age", "model2.joblib")
if not os.path.exists(os.path.dirname(path_model1)):
    os.makedirs(os.path.dirname(path_model1), exist_ok=True)
save_model(path_model1, model1)
save_model(path_model2, model2)
#%%
                                    # T-2,        T-1,          T, AGE
model1.predict_proba(np.array([0.06258286, 0.07668982, 0.07199386, 1  ]).reshape(1, -1))[-1]
model1.predict_proba(np.array([0.06258286, 0.07668982, 0.07199386, 2  ]).reshape(1, -1))[-1]
model1.predict_proba(np.array([0.06258286, 0.07668982, 0.07199386, 3  ]).reshape(1, -1))[-1]
#%%





























