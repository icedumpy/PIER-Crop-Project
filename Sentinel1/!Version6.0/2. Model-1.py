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

    Returns14
     
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
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210629"

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

# model_parameters = [
#     "drop_age",
#     'drop_min', 'drop_max',
#     'drop*bc_min', 'drop*bc_max',
#     'bc(t-2)_min', 'bc(t-1)_min', 'bc(t)_min', 'bc(t+1)_min', 'bc(t+2)_min',
# ]
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
fig, ax = plt.subplots()
ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model, df_plot_train[model_parameters], df_plot_train["label"], "Train", ax=ax)
ax, _, _, _, _, _ = plot_roc_curve(model, df_plot_test[model_parameters], df_plot_test["label"], "Test", color="r-", ax=ax)
ax = set_roc_plot_template(ax)
#%%
# Time for model#2 - Predict bin
#%%
df_plot_train_reg = df_plot.loc[(~(df_plot["ext_act_id"]%10).isin([8, 9])) & (df_plot["loss_ratio"] > 0)].copy()
df_plot_test_reg  = df_plot.loc[((df_plot["ext_act_id"]%10).isin([8, 9])) & (df_plot["loss_ratio"] > 0)].copy()
#%%
# Time for model#2 - Predict loss ratio
#%%
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df_plot_train_reg[model_parameters], df_plot_train_reg["loss_ratio"])
fig, ax = plot_hist_actual_predicted(df_plot_train_reg, df_plot_test_reg, reg)
fig.suptitle("Linear Regression")
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression
from sklearn.linear_model import PassiveAggressiveRegressor, TheilSenRegressor

list_names = [
    "Linear Regression", "Ridge", "RidgeCV", "Lasso", 
    "LassoCV", "LassoLarsCV", "LassoLarsIC", "ElasticNet",
    "Lars", "LassoLars", "OrthogonalMatchingPursuit", "BayesianRidge",
    "ARDRegression", "PassiveAggressiveRegressor", "TheilSenRegressor",
    "PolynomialFeatures(Degree=2)", "PolynomialFeatures(Degree=3)"
]

list_regressors  = [
    LinearRegression(), Ridge(), RidgeCV(), Lasso(),
    LassoCV(), LassoLarsCV(), LassoLarsIC(), ElasticNet(),
    Lars(), LassoLars(), OrthogonalMatchingPursuit(), BayesianRidge(),
    ARDRegression(), PassiveAggressiveRegressor(), TheilSenRegressor(),
    Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))]),
    Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False, n_jobs=7))]),
]

root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210824"
os.makedirs(os.path.join(root_save, strip_id), exist_ok=True)

dict_score = dict()
for name, reg in zip(list_names, list_regressors):
    plt.close("all")
    reg.fit(df_plot_train_reg[model_parameters], df_plot_train_reg["loss_ratio"])
    # Calculate r score, mae
    pred_train, pred_test, mae_train, mae_test, mse_train, mse_test, r2_train, r2_test = get_linear_score(df_plot_train_reg, df_plot_test_reg, reg)
    
    dict_score[name] = {
        'mae_train':mae_train, 'mae_test':mae_test, 'mse_train':mse_train, 'mse_test':mse_test,
        'r2_train':r2_train, 'r2_test':r2_test
    }
    # Plot hist (actual loss ratio, predicted loss ratio)
    fig, ax = plot_hist_actual_predicted(df_plot_train_reg, df_plot_test_reg, reg)    
    fig.suptitle(f"S{strip_id}, {name}")
    fig.savefig(os.path.join(root_save, strip_id, f"{name}-{strip_id}.png"), bbox_inches="tight")

    # Plot hist (actual loss ratio - predicted loss ratio)
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 9), sharex=False, sharey='row')
    # Train
    df_result = pd.DataFrame([df_plot_train_reg["loss_ratio"].to_numpy().T, np.clip(reg.predict(df_plot_train_reg[model_parameters]), 0, 1)], index=["Actual", "Predicted"]).T
    df_result = df_result.assign(**{"Actual-Predicted":df_result["Actual"]-df_result["Predicted"]})
    df_result = df_result.assign(digitized_loss_ratio = np.digitize(df_result["Actual"], [0, 0.25, 0.5, 0.75], right=True))
    for digitized_loss_ratio, df_results_grp, in df_result.groupby("digitized_loss_ratio"):
        ax[digitized_loss_ratio-1][0] = sns.histplot(
            df_results_grp["Actual-Predicted"], bins="auto", label=f"Bin:{digitized_loss_ratio}",
            stat="probability", kde=True, ax=ax[digitized_loss_ratio-1][0]
        )
        ax[digitized_loss_ratio-1][0].set_xlabel("")
        ax[digitized_loss_ratio-1][0].set_xlim(-1, 1)
    # Test
    df_result = pd.DataFrame([df_plot_test_reg["loss_ratio"].to_numpy().T, np.clip(reg.predict(df_plot_test_reg[model_parameters]), 0, 1)], index=["Actual", "Predicted"]).T
    df_result = df_result.assign(**{"Actual-Predicted":df_result["Actual"]-df_result["Predicted"]})
    df_result = df_result.assign(digitized_loss_ratio = np.digitize(df_result["Actual"], [0, 0.25, 0.5, 0.75], right=True))
    for digitized_loss_ratio, df_results_grp, in df_result.groupby("digitized_loss_ratio"):
        ax[digitized_loss_ratio-1][1] = sns.histplot(
            df_results_grp["Actual-Predicted"], bins="auto", label=f"Bin:{digitized_loss_ratio}",
            stat="probability", kde=True, ax=ax[digitized_loss_ratio-1][1]
        )
        ax[digitized_loss_ratio-1][1].set_xlabel("")
        ax[digitized_loss_ratio-1][1].set_xlim(-1, 1)
    ax[0][0].set_title("Train")
    ax[0][1].set_title("Test")
    [item.legend(loc=1) for item in ax.reshape(-1)]
    fig.supxlabel("Actual loss ratio - Predicted loss ratio")
    # fig.suptitle(f"{name}\nMAE(Train):{mae_train:.4f}, MAE(Test):{mae_test:.4f}\nR2(Train):{r2_train:.4f}, R2(Test):{r2_test:.4f}")
    fig.suptitle(f"S{strip_id}, {name}\nMAE(Train, Test):{mae_train:.4f}, {mae_test:.4f}\n$R^2$(Train, Test):{r2_train:.4f}, {r2_test:.4f}")
    fig.savefig(os.path.join(root_save, strip_id, f"{name}-diff-{strip_id}.png"), bbox_inches="tight")
df_score = pd.DataFrame(dict_score).T
df_score.to_csv(os.path.join(root_save, strip_id, f"results-{strip_id}.csv"))
    
    
    
    
    
#%%
pred_train, pred_test, mae_train, mae_test, mse_train, mse_test, r2_train, r2_test = get_linear_score(df_plot_train_reg, df_plot_test_reg, reg)
print(f"MAE(Train):{mae_train:.4f}, MAE(Test):{mae_test:.4f}")
print(f"MSE(Train):{mse_train:.4f}, MSE(Test):{mse_test:.4f}")
print(f"R2 (Train):{r2_train:.4f}, R2 (Test):{r2_test:.4f}")
#%%
fig, ax = plt.subplots()
sns.histplot(df_plot_train_reg["loss_ratio"]-pred_train, bins="auto", kde=True, stat="probability", ax=ax)
ax.set_xlabel("Actual loss ratio - Predicted loss ratio")
ax.set_xlim(-1, 1)
#%%
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 9), sharex=True)
ax[0][0].set_xlim(-1, 1)
sns.histplot(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.00) & (df_plot_train_reg["loss_ratio"] <= 0.25), "loss_ratio"]-reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.00) & (df_plot_train_reg["loss_ratio"] <= 0.25), model_parameters]), 
             bins="auto", kde=True, stat="probability", label="0.00 < loss_ratio <= 0.25", ax=ax[0][0])
sns.histplot(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.25) & (df_plot_train_reg["loss_ratio"] <= 0.50), "loss_ratio"]-reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.25) & (df_plot_train_reg["loss_ratio"] <= 0.50), model_parameters]), 
             bins="auto", kde=True, stat="probability", label="0.25 < loss_ratio <= 0.50", ax=ax[1][0])
sns.histplot(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.50) & (df_plot_train_reg["loss_ratio"] <= 0.75), "loss_ratio"]-reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.50) & (df_plot_train_reg["loss_ratio"] <= 0.75), model_parameters]), 
             bins="auto", kde=True, stat="probability", label="0.50 < loss_ratio <= 0.75", ax=ax[2][0])
sns.histplot(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.75) & (df_plot_train_reg["loss_ratio"] <= 1.00), "loss_ratio"]-reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.75) & (df_plot_train_reg["loss_ratio"] <= 1.00), model_parameters]), 
             bins="auto", kde=True, stat="probability", label="0.75 < loss_ratio <= 1.00", ax=ax[3][0])
ax[3][0].set_xlabel("Actual loss ratio - Predicted loss ratio")
[item.legend(loc=1) for item in ax[:, 0]]
#%%

for area in np.arange(0, 20000, 1000):
    df[df["TOTAL_ACTUAL_PLANT_AREA_IN_WA"].between(area, area+1000, inclusive="right")]
    break

























































#%%
path_sandbox = r"F:\CROP-PIER\CROP-WORK\Sandbox_ext_act_id.csv"
df_sandbox = pd.read_csv(path_sandbox)
df_selected = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df_selected = df_selected[df_selected["in_season_rice_f"] == 1]
df_selected = df_selected[(df_selected["DANGER_TYPE"] == "อุทกภัย") | (df_selected["DANGER_TYPE"]).isna()]
df_selected = df_selected[(df_selected["loss_ratio"] >= 0) & (df_selected["loss_ratio"] <= 1)]
df_selected = df_selected[df_selected["ext_act_id"].isin(df_sandbox["ext_act_id"])]
df_selected = pd.merge(df_selected, df_sandbox, how="inner", on="ext_act_id")
df_selected = convert_power_to_db(df_selected, columns_large)
df_selected = assign_sharp_drop(df_selected)
df_selected_plot = convert_pixel_level_to_plot_level(df_selected)
df_selected_plot = pd.merge(df_selected_plot, df_selected.groupby("ext_act_id").agg({"old_label":"mean", "new_label":"mean"}).reset_index(), on="ext_act_id")
df_selected_plot = df_selected_plot.assign(digitized_loss_ratio = np.digitize(df_selected_plot["loss_ratio"], [0, 0.25, 0.5, 0.75], right=True))
#%% Get diff reported days
list_diff = []
for _, df_selected_grp in df_selected.groupby("ext_act_id"):
    if df_selected_grp.iloc[0]["new_label"] == 1:
       list_diff.append((df_selected_grp.iloc[0]["DANGER_DATE"]-datetime.datetime.strptime(df_selected_grp.iloc[0]["Flood_date"], "%m/%d/%Y")).days)
fig, ax = plt.subplots()
sns.histplot(list_diff, bins="auto", ax=ax)
ax.set_xticks(np.arange(np.round(ax.get_xlim()[0]), np.round(ax.get_xlim()[1]), 5))
ax.set_xlabel("Reported_date - Ice_date")
fig.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210804\Report_diff.png", bbox_inches="tight")
#%% 
plt.close("all")
for digitized_loss_ratio, df_selected_plot_grp in df_selected_plot.groupby("digitized_loss_ratio"):
    cnf_matrix = confusion_matrix(df_selected_plot_grp["old_label"], df_selected_plot_grp["new_label"])
    plt.figure()
    sns.heatmap(cnf_matrix, annot=True, cbar=False)
    plt.xlabel("New label")
    plt.ylabel("Old label")
    plt.title(f"Bin: {digitized_loss_ratio}")
    plt.savefig(rf"F:\CROP-PIER\CROP-WORK\Presentation\20210804\Bin-{digitized_loss_ratio}.png", bbox_inches="tight")
cnf_matrix = confusion_matrix(df_selected_plot["old_label"], df_selected_plot["new_label"])
plt.figure()
sns.heatmap(cnf_matrix, annot=True, cbar=False)
plt.xlabel("New label")
plt.ylabel("Old label")
plt.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210804\New-label.png", bbox_inches="tight")
#%%
fig, ax = plt.subplots()
ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model, df_plot_train[model_parameters], df_plot_train["label"], "Train", ax=ax)
ax, _, _, _, _, _ = plot_roc_curve(model, df_plot_test[model_parameters], df_plot_test["label"], "Test", color="r-", ax=ax)
plot_roc_curve(model, df_selected_plot[model_parameters], df_selected_plot["new_label"], "Sandbox", color="g-", ax=ax)
ax.grid()
ax.legend(loc=4)
fig.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210804\ROC.png", bbox_inches="tight")
#%%
list_diff_2019 = []
list_diff_2020 = []
for (_, final_crop_year), df_selected_grp in df_selected.groupby(["ext_act_id", "final_crop_year"]):
    if df_selected_grp.iloc[0]["new_label"] == 1:
        if final_crop_year == 2019:
            list_diff_2019.append((df_selected_grp.iloc[0]["DANGER_DATE"]-datetime.datetime.strptime(df_selected_grp.iloc[0]["Flood_date"], "%m/%d/%Y")).days)
        elif final_crop_year == 2020:
            list_diff_2020.append((df_selected_grp.iloc[0]["DANGER_DATE"]-datetime.datetime.strptime(df_selected_grp.iloc[0]["Flood_date"], "%m/%d/%Y")).days)
plt.figure()
sns.histplot(list_diff_2019)
plt.xlabel("Reported_date - Ice_date")
plt.title("2019")
plt.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210804\Diff_date_2019.png", bbox_inches="tight")

plt.figure()
sns.histplot(list_diff_2020)
plt.xlabel("Reported_date - Ice_date")
plt.title("2020")
plt.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210804\Diff_date_2020.png", bbox_inches="tight")
#%%
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
#%% Check result
threshold = get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr=0.2)
df_plot_test = df_plot_test.assign(predict_proba = model.predict_proba(df_plot_test[model_parameters])[:, 1])
df_plot_test.loc[df_plot_test["predict_proba"] == 0, "predict_proba"] = 0.01
df_plot_test.loc[df_plot_test["predict_proba"] == 0, "predict_proba"] = 0.99
df_plot_test = df_plot_test.assign(bce = -(df_plot_test["label"]*np.log(df_plot_test["predict_proba"]))-((1-df_plot_test["label"])*np.log(1-df_plot_test["predict_proba"])))
df_plot_test = df_plot_test.sort_values(by="bce", ascending=False)
#%% Check FA
df_plot_test_fa = df_plot_test[(df_plot_test["label"] == 0) & (df_plot_test["predict_proba"] >= threshold)]
df_plot_test_ms = df_plot_test[(df_plot_test["label"] == 1) & (df_plot_test["predict_proba"] <= threshold)]
#%%
for _, sample in tqdm(df_plot_test_fa.iterrows(), total=len(df_plot_test_fa)):
    plt.close("all")
    ext_act_id = sample.ext_act_id
    predict_proba = sample.predict_proba
    bce = sample.bce
    sample = df[df["ext_act_id"] == ext_act_id]
    fig, ax = initialize_plot()
    ax.plot(sample[columns_large].T)
    ax.axvline(sample.drop_column.iloc[0], linestyle="--")
    ax.grid()
    fig.suptitle(f"ext_act_id: {int(ext_act_id)}\nProb: {predict_proba:.2f}, BCE: {bce:.2f}")
    fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210714\FA", f"{bce:.2f}_{int(ext_act_id)}.png"), bbox_inches="tight")

for _, sample in tqdm(df_plot_test_ms.iterrows(), total=len(df_plot_test_ms)):
    plt.close("all")
    ext_act_id = sample.ext_act_id
    predict_proba = sample.predict_proba
    bce = sample.bce
    loss_ratio = sample.loss_ratio
    
    sample = df[df["ext_act_id"] == ext_act_id]
    fig, ax = initialize_plot()
    ax.plot(sample[columns_large].T)
    ax.axvline(sample.drop_column.iloc[0], linestyle="--")
    ax.grid()
    fig.suptitle(f"ext_act_id: {int(ext_act_id)}, loss_ratio: {loss_ratio:.2f}\nProb: {predict_proba:.2f}, BCE: {bce:.2f}")
    fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210714\MS", f"{bce:.2f}_{int(ext_act_id)}.png"), bbox_inches="tight")
plt.figure()
sns.histplot(df_plot_test_fa["bce"], stat="probability", cumulative=True, element="step", fill=False)
#%% Step2: If nonflood -> skip. Otherwise -> Regression model
#%%


plt.figure()
sns.histplot(reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.00) & (df_plot_train_reg["loss_ratio"] <= 0.25), model_parameters]), bins="auto")
plt.figure()

sns.histplot(reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.25) & (df_plot_train_reg["loss_ratio"] <= 0.50), model_parameters]), bins="auto")
plt.figure()

sns.histplot(reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.50) & (df_plot_train_reg["loss_ratio"] <= 0.75), model_parameters]), bins="auto")
plt.figure()

sns.histplot(reg.predict(df_plot_train_reg.loc[(df_plot_train_reg["loss_ratio"] > 0.75) & (df_plot_train_reg["loss_ratio"] <= 1.00), model_parameters]), bins="auto")



















































