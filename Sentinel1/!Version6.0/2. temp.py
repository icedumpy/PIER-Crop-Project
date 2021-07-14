import os
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
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
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210629"

# Classifier hyperparameters
# Number of trees in random forest
n_estimators = [100, 150, 300]
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

# Convert pixel-level to plot-level
print("Converting to plot level")
df_plot = convert_pixel_level_to_plot_level(df)

# Digitize loss ratio
df_plot = df_plot.assign(digitized_loss_ratio = np.digitize(df_plot["loss_ratio"], [0, 0.25, 0.5, 0.75], right=True))

# Label for flood or nonflood
df_plot.loc[df_plot["loss_ratio"] == 0, "label"] = 0
df_plot.loc[df_plot["loss_ratio"] != 0, "label"] = 1
#%%
# model_parameters = [
#     "drop_age",
#     'drop_min', 'drop_max', 'drop_p25', 'drop_p50', 'drop_p75',
#     'drop*bc_min', 'drop*bc_max', 'drop*bc_p25', 'drop*bc_p50', 'drop*bc_p75',
#     'bc(t-2)_min', 'bc(t-1)_min', 'bc(t)_min', 'bc(t+1)_min', 'bc(t+2)_min',
#     'bc(t-2)_max', 'bc(t-1)_max', 'bc(t)_max', 'bc(t+1)_max', 'bc(t+2)_max',
#     'bc(t-2)_p25', 'bc(t-1)_p25', 'bc(t)_p25', 'bc(t+1)_p25', 'bc(t+2)_p25',
#     'bc(t-2)_p50', 'bc(t-1)_p50', 'bc(t)_p50', 'bc(t+1)_p50', 'bc(t+2)_p50',
#     'bc(t-2)_p75', 'bc(t-1)_p75', 'bc(t)_p75', 'bc(t+1)_p75', 'bc(t+2)_p75',
# ]

model_parameters = [
    "drop_age",
    'drop_min', 'drop_max',
    'drop*bc_min', 'drop*bc_max',
    'bc(t-2)_min', 'bc(t-1)_min', 'bc(t)_min', 'bc(t+1)_min', 'bc(t+2)_min',
]
#%% Step1: Classify for flood or nonflood
df_plot_train = df_plot.loc[~(df_plot["ext_act_id"]%10).isin([8, 9])].copy()
df_plot_test  = df_plot.loc[(df_plot["ext_act_id"]%10).isin([8, 9])].copy()

plt.close("all")
# =============================================================================
# Train model (First time)
# =============================================================================
model = RandomizedSearchCV(estimator=RandomForestClassifier(),
                           param_distributions=random_grid,
                           n_iter=20,
                           cv=5,
                           verbose=2,
                           random_state=42,
                           n_jobs=-1,
                           scoring='roc_auc'
                           )
# Fine-tuning model
model.fit(df_plot_train[model_parameters], df_plot_train["label"])
model = model.best_estimator_

# Plot results #1
ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model, df_plot_train[model_parameters], df_plot_train["label"], "Train")
ax, _, _, _, _, _ = plot_roc_curve(model, df_plot_test[model_parameters], df_plot_test["label"], "Test", color="r-", ax=ax)
ax = set_roc_plot_template(ax)

# # Calculate BCE and clean p90
# df_plot_train = df_plot_train.assign(predict_proba = model.predict_proba(df_plot_train[model_parameters])[:, 1])
# df_plot_train = df_plot_train.assign(bce = -(df_plot_train["label"]*np.log(df_plot_train["predict_proba"]))-((1-df_plot_train["label"])*np.log(1-df_plot_train["predict_proba"])))

# threshold_bce_0 = df_plot_train.loc[df_plot_train["label"] == 0, "bce"].quantile(0.9)
# threshold_bce_1 = df_plot_train.loc[df_plot_train["label"] == 1, "bce"].quantile(0.9)

# # Cleaned data
# df_plot_train_cleaned = pd.concat([
#     df_plot_train.loc[(df_plot_train["label"] == 0) & (df_plot_train["bce"] <= threshold_bce_0)],
#     df_plot_train.loc[(df_plot_train["label"] == 1) & (df_plot_train["bce"] <= threshold_bce_1)]
# ], ignore_index=True)

# # =============================================================================
# # Train model (Second time w/ cleaner data)
# # =============================================================================
# model = RandomizedSearchCV(estimator=RandomForestClassifier(),
#                            param_distributions=random_grid,
#                            n_iter=20,
#                            cv=5,
#                            verbose=2,
#                            random_state=42,
#                            n_jobs=-1,
#                            scoring='roc_auc'
#                            )
# # Fine-tuning model
# model.fit(df_plot_train_cleaned[model_parameters], df_plot_train_cleaned["label"])
# model = model.best_estimator_

# # Plot results #2
# ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model, df_plot_train_cleaned[model_parameters], df_plot_train_cleaned["label"], "Train")
# ax, _, _, _, _, _ = plot_roc_curve(model, df_plot_test[model_parameters], df_plot_test["label"], "Test", color="r-", ax=ax)
# ax = set_roc_plot_template(ax)

plt.figure()
plt.barh([a[1] for a in sorted(zip(model.feature_importances_, model_parameters))], [100*a[0] for a in sorted(zip(model.feature_importances_, model_parameters))])
#%% Check result
threshold = get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr=0.2)
df_plot_test = df_plot_test.assign(predict_proba = model.predict_proba(df_plot_test[model_parameters])[:, 1])
df_plot_test = df_plot_test.assign(bce = -(df_plot_test["label"]*np.log(df_plot_test["predict_proba"]))-((1-df_plot_test["label"])*np.log(1-df_plot_test["predict_proba"])))
df_plot_test = df_plot_test.sort_values(by="bce", ascending=False)
#%% Check FA
df_plot_test_fa = df_plot_test[(df_plot_test["label"] == 0) & (df_plot_test["predict_proba"] >= threshold)]
#%%
plt.close("all")
for _, sample in df_plot_test_fa.tail().iterrows():
    ext_act_id = sample.ext_act_id
    sample = df[df["ext_act_id"] == ext_act_id]
    fig, ax = initialize_plot()
    ax.plot(sample[columns_large].T)
#%%
sns.histplot(df_plot_test_fa["bce"], stat="probability", cumulative=True, element="step", fill=False)
#%% Step2: If nonflood -> skip. Otherwise -> Regression model
#%%
#%%













