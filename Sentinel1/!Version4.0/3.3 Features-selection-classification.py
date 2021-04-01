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
from icedumpy.io_tools import save_model
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
    df.loc[df["loss_ratio"] >= 0.8, "label"] = 1
    df["label"] = df["label"].astype("uint8")
    return df


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
# %%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210317\Fig"

path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
strip_id = "304"
# %%
# for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
print(strip_id)
# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]

# Load df pixel value
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(
    root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df.drop(columns="t30")  # Column "t30" is already out of season
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), 2 *
         len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]

# Define columns' group name
columns = df.columns[:30]
columns_age1 = [f"t{i}" for i in range(0, 7)]  # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)]  # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)]  # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)]  # 120-179

# Convert power to db
df = convert_power_to_db(df, columns)

# Drop duplicates
df = df.drop_duplicates(subset=columns)

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

# Merge photo sensitivity
df = pd.merge(df, df_rice_code, on="BREED_CODE", how="inner")

df = df.sample(frac=1.0)
# %%
model_parameters_1 = ['median', 'median(age1)', 'median(age2)', 'median(age3)', 'median(age4)',
                      'median-min', 'median-min(age1)', 'median-min(age2)', 'median-min(age3)', 'median-min(age4)',
                      'min', 'min(age1)', 'min(age2)', 'min(age3)', 'min(age4)',
                      'max-min', 'max-min(age1)', 'max-min(age2)', 'max-min(age3)', 'max-min(age4)',
                      'area-under-median', 'area-under-median(age1)', 'area-under-median(age2)',
                      'area-under-median(age3)', 'area-under-median(age4)',
                      'count-under-median', 'count-under-median(age1)', 'count-under-median(age2)',
                      'count-under-median(age3)', 'count-under-median(age4)',
                      'photo_sensitive_f']

model_parameters_2 = ['median', 'median(min)',
                      'median-min', 'median-min(max)',
                      'min(age1)', 'min(age2)', 'min(age3)', 'min(age4)',
                      'max-min', 'max-min(max)',
                      'area-under-median', 'area-under-median(max)',
                      'count-under-median', 'count-under-median(max)',
                      'photo_sensitive_f']
# %%
# Model hyperparameters
# Number of trees in random forest
n_estimators = [20, 50, 100]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 50, num=3)]
min_samples_split = [3, 5, 10]
min_samples_leaf = [3, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)
# %%
x = df[model_parameters_1].values
y = df["label"].values
# %%
model = RandomizedSearchCV(estimator=RandomForestClassifier(),
                           param_distributions=random_grid,
                           n_iter=10,
                           cv=5,
                           verbose=2,
                           random_state=42,
                           n_jobs=-1,
                           scoring='roc_auc')
# Fit the random search model
model.fit(x, y)
model = model.best_estimator_

fig, ax = plt.subplots(figsize=(16, 9))
ax, _, _, _, _, _ = plot_roc_curve(model, x, y, color="g-", label="all", ax=ax)
ax = set_roc_plot_template(ax)
#%% 
# Calculate bce loss
df = df.assign(predict_proba = model.predict_proba(x)[:, 1])
df.loc[df["predict_proba"] == 0, "predict_proba"] = 1e-7
df.loc[df["predict_proba"] == 1, "predict_proba"] = 1-1e-7
df = df.assign(bce_loss=-(df["label"]*np.log(df["predict_proba"]))-((1-df["label"])*np.log(1-df["predict_proba"])))

# Drop high bce_loss data (95th percentile)
df = df[df["bce_loss"] < df["bce_loss"].quantile(0.95)]
#%%
# Now we have higher data quality
x_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), model_parameters_1].values
x_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), model_parameters_1].values

y_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), "label"].values
y_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), "label"].values
#%%
# Retrain model
model = RandomizedSearchCV(estimator=RandomForestClassifier(),
                           param_distributions=random_grid,
                           n_iter=10,
                           cv=5,
                           verbose=2,
                           random_state=42,
                           n_jobs=-1,
                           scoring='roc_auc')
# Fit the random search model
model.fit(x_train, y_train)
model = model.best_estimator_

# Re-train with best_estimators_ (Should be the same as previous best_estimator but want to make sure)
model.fit(x_train, y_train)
#%%
fig, ax = plt.subplots(figsize=(16, 9))
ax, _, _, _, _, _ = plot_roc_curve(model, x_train, y_train, color="g-", label="trian", ax=ax)
ax, _, _, _, _, _ = plot_roc_curve(model, x_test, y_test, color="b-", label="test", ax=ax)
ax = set_roc_plot_template(ax)

fig, ax = plt.subplots(figsize=(16, 9))
ax, _, _, _, _, _ = plot_precision_recall_curve(model, x_train, y_train, color="g-", label="trian", ax=ax)
ax, _, _, _, _, _ = plot_precision_recall_curve(model, x_test, y_test, color="b-", label="test", ax=ax)
ax = set_roc_plot_template(ax)
#%%
folder_model = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season"
path_model = os.path.join(folder_model, f"{strip_id}.joblib")
save_model(path_model, model)
#%%