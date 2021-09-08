import os
import datetime
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import seaborn as sns
from pprint import pprint
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
#%%
@jit(nopython=True)
def interp_numba(arr_ndvi):
    '''
    Interpolate an array in both directions using numba.
    (From     'Tee+)

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

def assign_sharp_drop(df):
    df = df.copy()
    
    # Loop for each group (group by ext_act_id)
    list_df = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        # Find which "period" (1, 2, or 3) gives the most extreme diff*backscatter
        periods = int(np.argmax([
            (df_grp[columns_model].diff(periods=1, axis=1)*df_grp[columns_model]).max(axis=1).max(), 
            (df_grp[columns_model].diff(periods=2, axis=1)*df_grp[columns_model]).max(axis=1).max(),
            (df_grp[columns_model].diff(periods=3, axis=1)*df_grp[columns_model]).max(axis=1).max()
        ])+1)
        
        # Find which column
        drop = df_grp[columns_model].diff(periods=periods, axis=1)
        coef = drop*df_grp[columns_model]
        flood_column = coef.max().idxmax()
        
        # Add drop value
        df_grp["drop"] = drop[flood_column]
        
        # Add the most extreme diff*backscatter
        df_grp["drop*bc"] = coef.max(axis=1).values
        
        # Add sharp-drop column
        flood_column = int(flood_column[1:])
        df_grp["drop_column"] = f"t{flood_column}"
        
        # Extract data (-2, +2)
        df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]] = df_grp[[f"t{i}" for i in range(flood_column-2, flood_column+3)]].values
        list_df.append(df_grp)
        
    # Concat and return
    df = pd.concat(list_df, ignore_index=True)
    return df

def convert_pixel_level_to_plot_level(df):
    df = df.copy()
    
    list_dict_plot = []
    for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
        loss_ratio = df_grp.iloc[0]["loss_ratio"]

        # Agg parameters
        drop_min = df_grp["drop"].min()
        drop_max = df_grp["drop"].max()
        drop_p25 = df_grp["drop"].quantile(0.25)
        drop_p50 = df_grp["drop"].quantile(0.50)
        drop_p75 = df_grp["drop"].quantile(0.75)
        drop_bc_min = df_grp["drop*bc"].min()
        drop_bc_max = df_grp["drop*bc"].max()
        drop_bc_p25 = df_grp["drop*bc"].quantile(0.25)
        drop_bc_p50 = df_grp["drop*bc"].quantile(0.50)
        drop_bc_p75 = df_grp["drop*bc"].quantile(0.75)
        bc_min = df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]].min(axis=0).values
        bc_max = df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]].max(axis=0).values
        bc_p25 = df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]].quantile(0.25, axis=0).values
        bc_p50 = df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]].quantile(0.50, axis=0).values
        bc_p75 = df_grp[["bc(t-2)", "bc(t-1)", "bc(t)", "bc(t+1)", "bc(t+2)"]].quantile(0.75, axis=0).values
        drop_column = df_grp.iloc[0]["drop_column"]
        if drop_column in columns_age2:
            drop_age = 1
        elif drop_column in columns_age3:
            drop_age = 2
        elif drop_column in columns_age4:
            drop_age = 3
            
        # Create dict of parameters
        dict_plot = {
            "ext_act_id":ext_act_id,
            "drop_age":drop_age,
            "drop_min":drop_min,
            "drop_max":drop_max,
            "drop_p25":drop_p25,
            "drop_p50":drop_p50,
            "drop_p75":drop_p75,
            "drop*bc_min":drop_bc_min,
            "drop*bc_max":drop_bc_max,
            "drop*bc_p25":drop_bc_p25,
            "drop*bc_p50":drop_bc_p50,
            "drop*bc_p75":drop_bc_p75,
            "bc(t-2)_min":bc_min[0],
            "bc(t-1)_min":bc_min[1],
            "bc(t)_min"  :bc_min[2],
            "bc(t+1)_min":bc_min[3],
            "bc(t+2)_min":bc_min[4],
            "bc(t-2)_max":bc_max[0],
            "bc(t-1)_max":bc_max[1],
            "bc(t)_max"  :bc_max[2],
            "bc(t+1)_max":bc_max[3],
            "bc(t+2)_max":bc_max[4],
            "bc(t-2)_p25":bc_p25[0],
            "bc(t-1)_p25":bc_p25[1],
            "bc(t)_p25"  :bc_p25[2],
            "bc(t+1)_p25":bc_p25[3],
            "bc(t+2)_p25":bc_p25[4],
            "bc(t-2)_p50":bc_p50[0],
            "bc(t-1)_p50":bc_p50[1],
            "bc(t)_p50"  :bc_p50[2],
            "bc(t+1)_p50":bc_p50[3],
            "bc(t+2)_p50":bc_p50[4],
            "bc(t-2)_p75":bc_p75[0],
            "bc(t-1)_p75":bc_p75[1],
            "bc(t)_p75"  :bc_p75[2],
            "bc(t+1)_p75":bc_p75[3],
            "bc(t+2)_p75":bc_p75[4],
            "loss_ratio":loss_ratio
        }
        
        # Append dict to list
        list_dict_plot.append(dict_plot)
        
    # Create dataframe and return
    df_plot = pd.DataFrame(list_dict_plot)
    return df_plot

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def get_linear_score(df_plot_train_reg, df_plot_test_reg, reg):
    pred_train = np.clip(reg.predict(df_plot_train_reg[model_parameters]), 0, 1)
    pred_test  = np.clip(reg.predict(df_plot_test_reg[model_parameters]), 0, 1)
    
    mae_train = mean_absolute_error(df_plot_train_reg["loss_ratio"], pred_train)
    mae_test  = mean_absolute_error(df_plot_test_reg["loss_ratio"], pred_test)
    
    mse_train = mean_squared_error(df_plot_train_reg["loss_ratio"], pred_train)
    mse_test  = mean_squared_error(df_plot_test_reg["loss_ratio"], pred_test)
    
    r2_train = r2_score(df_plot_train_reg["loss_ratio"], pred_train)
    r2_test  = r2_score(df_plot_test_reg["loss_ratio"], pred_test)
    
    return pred_train, pred_test, mae_train, mae_test, mse_train, mse_test, r2_train, r2_test

# Not work (distribution of output is completely changed)
def sigmoid(x):
    return 1/(1+np.power(np.e, -x))

def plot_hist_actual_predicted(df_plot_train_reg, df_plot_test_reg, reg):
    fig, ax = plt.subplots(ncols=2, sharey=True)
    df_result = pd.DataFrame([df_plot_train_reg["loss_ratio"].to_numpy().T, np.clip(reg.predict(df_plot_train_reg[model_parameters]), 0, 1)], index=["Actual", "Predicted"]).T
    ax[0] = sns.histplot(
        data=df_result.melt(var_name='Type', value_name='Loss ratio'), x="Loss ratio", hue="Type",
        stat="probability", element="step", ax=ax[0]
    )
    ax[0].set_title("Train")
    
    df_result = pd.DataFrame([df_plot_test_reg["loss_ratio"].to_numpy().T, np.clip(reg.predict(df_plot_test_reg[model_parameters]), 0, 1)], index=["Actual", "Predicted"]).T
    ax[1] = sns.histplot(
        data=df_result.melt(var_name='Type', value_name='Loss ratio'), x="Loss ratio", hue="Type",
        stat="probability", element="step", ax=ax[1]
    )
    ax[1].set_title("Test")
    return fig, ax
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210903"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"

# Classifier hyperparameters
# Number of trees in random forest
n_estimators = [100, 250, 500]
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
columns_age1 = [f"t{i}" for i in range(0, 7)]   # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)] # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)] # 120-179

columns_model = columns_age1[-1:]+columns_age2+columns_age3+columns_age4
#%%
# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]
#%%
strip_id = "304"
os.makedirs(os.path.join(root_save, strip_id), exist_ok=True)

df = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[df["in_season_rice_f"] == 1]
df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]

# Samping 
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), 
                                                len(df.loc[df["loss_ratio"] > 0, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] > 0)]

# Convert power to dB
print("Converting to dB")
df = convert_power_to_db(df, columns_large)

# Assign sharp drop
print("Assigning sharp drop")
df = assign_sharp_drop(df)

# Merge photo sensitivity
print("Merging photo sensitivity")
df = pd.merge(df, df_rice_code, on="BREED_CODE", how="inner")

# Convert pixel-level to plot-level
print("Converting to plot level")
df_plot = convert_pixel_level_to_plot_level(df)

# Digitize loss ratio
df_plot = df_plot.assign(digitized_loss_ratio = np.digitize(df_plot["loss_ratio"], [0, 0.25, 0.5, 0.75], right=True))

# Label for flood or nonflood
df_plot.loc[df_plot["loss_ratio"] == 0, "label"] = 0
df_plot.loc[df_plot["loss_ratio"] != 0, "label"] = 1
#%%
model_parameters = [
    "drop_age",
    'drop_min', 'drop_max', 'drop_p25', 'drop_p50', 'drop_p75',
    'drop*bc_min', 'drop*bc_max', 'drop*bc_p25', 'drop*bc_p50', 'drop*bc_p75',
    'bc(t-2)_min', 'bc(t-1)_min', 'bc(t)_min', 'bc(t+1)_min', 'bc(t+2)_min',
    'bc(t-2)_max', 'bc(t-1)_max', 'bc(t)_max', 'bc(t+1)_max', 'bc(t+2)_max',
    'bc(t-2)_p25', 'bc(t-1)_p25', 'bc(t)_p25', 'bc(t+1)_p25', 'bc(t+2)_p25',
    'bc(t-2)_p50', 'bc(t-1)_p50', 'bc(t)_p50', 'bc(t+1)_p50', 'bc(t+2)_p50',
    'bc(t-2)_p75', 'bc(t-1)_p75', 'bc(t)_p75', 'bc(t+1)_p75', 'bc(t+2)_p75',
]
df_plot_train = df_plot.loc[~(df_plot["ext_act_id"]%10).isin([8, 9])].copy()
df_plot_test  = df_plot.loc[(df_plot["ext_act_id"]%10).isin([8, 9])].copy()
#%%
# Label for flood or nonflood
df.loc[df["loss_ratio"] == 0, "label"] = "normal"
df.loc[df["loss_ratio"] != 0, "label"] = "flood"
#%%
# Fetures hist 
plt.close("all")
for feature in ['drop', 'drop*bc', 'bc(t-2)', 'bc(t-1)', 'bc(t)', 'bc(t+1)', 'bc(t+2)']:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue="label", ax=ax)
    try:
        fig.savefig(os.path.join(root_save, f"{strip_id}_{feature}.png"), bbox_inches="tight")
    except:
        fig.savefig(os.path.join(root_save, f"{strip_id}_{feature.replace('*', 'x')}.png"), bbox_inches="tight")
#%%
# Loss ratio hist
plt.close("all")
fig, ax = plt.subplots()
sns.histplot(df.loc[(df["loss_ratio"] > 0) & (df["loss_ratio"] <= 1), "loss_ratio"], stat="probability", ax=ax)
ax.set_title("0 < Loss ratio <= 1")
fig.savefig(os.path.join(root_save, f"{strip_id}_loss_ratio(0,1].png"), bbox_inches="tight")

fig, ax= plt.subplots()
sns.histplot(df.loc[(df["loss_ratio"] > 0) & (df["loss_ratio"] <= 1), "loss_ratio"], stat="probability",
             cumulative=True, ax=ax)
ax.set_title("0 < Loss ratio <= 1")
fig.savefig(os.path.join(root_save, f"{strip_id}_loss_ratio(0,1]_cumulative.png"), bbox_inches="tight")

fig, ax = plt.subplots()
sns.histplot(df.loc[(df["loss_ratio"] > 0) & (df["loss_ratio"] <  1), "loss_ratio"], stat="probability", ax=ax)
ax.set_title("0 < Loss ratio <  1")
fig.savefig(os.path.join(root_save, f"{strip_id}_loss_ratio(0,1).png"), bbox_inches="tight")

#%%
# Rice age hist
plt.close("all")
fig, ax = plt.subplots()
sns.histplot(data=df_plot[df_plot["loss_ratio"] != 0], x="drop_min", hue="drop_age", kde=True,
             palette="tab10", ax=ax)
fig.savefig(os.path.join(root_save, f"{strip_id}_drop_min_age.png"), bbox_inches="tight")

fig, ax = plt.subplots()
sns.histplot(data=df_plot[df_plot["loss_ratio"] != 0], x="bc(t)_min", hue="drop_age", kde=True,
             palette="tab10", ax=ax)
fig.savefig(os.path.join(root_save, f"{strip_id}_bc(t)_min_age.png"), bbox_inches="tight")

fig, ax = plt.subplots()
sns.histplot(data=df_plot[df_plot["loss_ratio"] != 0], x="drop*bc_max", hue="drop_age", kde=True,
             palette="tab10", ax=ax)
fig.savefig(os.path.join(root_save, f"{strip_id}_dropxbc_max_age.png"), bbox_inches="tight")
#%% Show time-series
for ext_act_id, df_grp in df[df["label"] == "normal"].groupby("ext_act_id"):
    plt.close("all")
    
    fig, ax = initialize_plot()
    ax.plot(df_grp[columns_large].values.T)
    ax.axvline(int(df_grp.iloc[0]["drop_column"][1:]), linestyle="--", color="red")
    ax.grid()
    plt.pause(1)
#%%
# Photosentivity
plt.close("all")
plt.bar(['Non-photosensitive', 'Photosensitive'], df.groupby("ext_act_id").agg({"photo_sensitive_f":"mean"}).value_counts(normalize=True).sort_index())
for i, value in enumerate(df.groupby("ext_act_id").agg({"photo_sensitive_f":"mean"}).value_counts(normalize=True).sort_index()):
    plt.text(i, value+0.01, f"{100*value:.2f}%", horizontalalignment="center")
plt.savefig(os.path.join(root_save, f"{strip_id}_photosensitivity_ratio.png"), bbox_inches="tight")
#%%
# min Hist
df = df.assign(**{"min":df[columns_model].min(axis=1)})
sns.histplot(data=df, x="min", hue="label")
#%%
# flood month hist
df_temp = pd.DataFrame(df[df["label"] == "flood"].groupby("danger_year")["danger_month"].value_counts())
df_temp.columns = ["count"]
df_temp = df_temp.reset_index()
df_temp = df_temp.astype("uint32")
sns.barplot(data=df_temp, x="danger_month", y="count", hue="danger_year")
plt.savefig(os.path.join(root_save, f"{strip_id}_danger_freq.png"), bbox_inches="tight")



