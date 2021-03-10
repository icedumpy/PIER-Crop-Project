import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
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
        arr_ndvi_row_not_nan_idx = np.argwhere(~np.isnan(arr_ndvi_row)).flatten()
        if len(arr_ndvi_row_not_nan_idx) > 0:
            arr_ndvi_row_not_nan_values = arr_ndvi_row[arr_ndvi_row_not_nan_idx]
            arr_ndvi[n_row] = np.interp(arr_ndvi_row_idx, arr_ndvi_row_not_nan_idx, arr_ndvi_row_not_nan_values)

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
    return df

def initialize_plot(ylim=(-20, -5)):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw group age
    ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
    ax.axvspan(6.5, 15.5, alpha=0.2, color='green')
    ax.axvspan(15.5, 30, alpha=0.2, color='yellow')
    [ax.axvline(i, color="black") for i in [6.5, 15.5]]
    
    # Add group descriptions
    ax.text(2.5, ylim[-1]+0.25, "0-40 days")
    ax.text(10, ylim[-1]+0.25, "40-90 days")
    ax.text(22, ylim[-1]+0.25, "90+ days")

    # Set y limits
    ax.set_ylim(ylim)

    # Add more ticks
    ax.set_xticks(range(31))
    ax.set_yticks(np.arange(*ylim))

    return fig, ax

@jit(nopython=True)
def get_area_under_median(arr_min_index, arr_value_under_median):
    arr_area_under_median = np.zeros_like(arr_min_index, dtype=np.float32)
    for i, (index_min, value_under_median)in enumerate(zip(arr_min_index, arr_value_under_median)):
        arr_group = np.zeros((arr_value_under_median.shape[-1]))
        group = 1
        for j, value in enumerate(value_under_median):
            if value > 0:
                arr_group[j] = group
            else:
                group+=1
        area_under_median = value_under_median[np.where(arr_group == arr_group[index_min])[0]].sum()
        arr_area_under_median[i] = area_under_median
    return arr_area_under_median
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210224\Fig"

path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
strip_id = "304"
#%%
# for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
print(strip_id)
# Load df rice code
df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]

# Load df pixel value
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df.drop(columns="t30") # Column "t30" is already out of season
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), 2*len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
#%%
# Define columns' group name
columns = df.columns[:30]
columns_age1 = [f"t{i}" for i in range(0, 7)] # 0-41
columns_age2 = [f"t{i}" for i in range(7, 15)] # 42-89
columns_age3 = [f"t{i}" for i in range(15, 20)] # 90-119
columns_age4 = [f"t{i}" for i in range(20, 30)] # 120-179
#%%
# Convert power to db
df = convert_power_to_db(df, columns)

# Drop duplicates
df = df.drop_duplicates(subset=columns)
#%%
# Assign median
df = df.assign(**{"median":df[columns].median(axis=1),
                  "median(age1)":df[columns_age1].median(axis=1),
                  "median(age2)":df[columns_age2].median(axis=1),
                  "median(age3)":df[columns_age3].median(axis=1),
                  "median(age4)":df[columns_age4].median(axis=1),
                  })

# Assign median - min 
df = df.assign(**{"median-min":df["median"]-df[columns].min(axis=1),
                  "median-min(age1)":df["median(age1)"]-df[columns_age1].min(axis=1),
                  "median-min(age2)":df["median(age2)"]-df[columns_age2].min(axis=1),
                  "median-min(age3)":df["median(age3)"]-df[columns_age3].min(axis=1),
                  "median-min(age4)":df["median(age4)"]-df[columns_age4].min(axis=1),
                  })

# Assign sum of cunsecutive values under median (around min value)
df = df.assign(**{"area-under-median(age1)" : get_area_under_median(np.argmax(df[columns_age1].eq(df[columns_age1].min(axis=1), axis=0).values, axis=1), (-df[columns_age1].sub(df["median(age1)"], axis=0)).clip(lower=0, upper=None).values),
                  "area-under-median(age2)" : get_area_under_median(np.argmax(df[columns_age2].eq(df[columns_age2].min(axis=1), axis=0).values, axis=1), (-df[columns_age2].sub(df["median(age2)"], axis=0)).clip(lower=0, upper=None).values),
                  "area-under-median(age3)" : get_area_under_median(np.argmax(df[columns_age3].eq(df[columns_age3].min(axis=1), axis=0).values, axis=1), (-df[columns_age3].sub(df["median(age3)"], axis=0)).clip(lower=0, upper=None).values),
                  "area-under-median(age4)" : get_area_under_median(np.argmax(df[columns_age4].eq(df[columns_age4].min(axis=1), axis=0).values, axis=1), (-df[columns_age4].sub(df["median(age4)"], axis=0)).clip(lower=0, upper=None).values)
                  })

# Assign min value 
df = df.assign(**{"min(age1)":df[columns_age1].min(axis=1),
                  "min(age2)":df[columns_age2].min(axis=1),
                  "min(age3)":df[columns_age3].min(axis=1),
                  "min(age4)":df[columns_age4].min(axis=1),
                  })

# Assign max-min value 
df = df.assign(**{"max-min":df[columns].max(axis=1)-df[columns].min(axis=1),
                  "max-min(age1)":df[columns_age1].max(axis=1)-df[columns_age1].min(axis=1),
                  "max-min(age2)":df[columns_age2].max(axis=1)-df[columns_age2].min(axis=1),
                  "max-min(age3)":df[columns_age3].max(axis=1)-df[columns_age3].min(axis=1),
                  "max-min(age4)":df[columns_age4].max(axis=1)-df[columns_age4].min(axis=1),
                  })

# Merge photo sensitivity
df = pd.merge(df, df_rice_code, on="BREED_CODE", how="inner")

df_nonflood = df[df["label"] == 0]
df_flood = df[df["label"] == 1]

# Add flood column
# df_flood = df_flood.assign(flood_column = (df_flood["START_DATE"]-df_flood["final_plant_date"]).dt.days//6)
#%%
# plt.close('all')
# sns.histplot(data=df, x="median", hue="label")
#%%
x_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), df.columns[-24:]].values
y_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), "label"].values
x_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]),  df.columns[-24:]].values
y_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), "label"].values
#%%
# model = RandomForestClassifier(n_jobs=-1)
model = RandomForestClassifier(min_samples_leaf=5, max_depth=10, min_samples_split=10,
                               verbose=0, n_jobs=-1, random_state=42)
model.fit(x_train, y_train)

plt.close('all')
fig, ax = plt.subplots(figsize=(16, 9))
plot_roc_curve(model, x_train, y_train, label="train", color="b-", ax=ax)       
plot_roc_curve(model, x_test, y_test, label="test", color="g-", ax=ax)
ax = set_roc_plot_template(ax)
ax.set_title(f'ROC Curve: {strip_id}\nAll_touched(False), Tier{(1,)}\nTrain samples: Flood:{np.bincount(y_train.astype(int))[1]:,}, Non-Flood:{np.bincount(y_train.astype(int))[0]:,}\nTest samples: Flood:{np.bincount(y_test.astype(int))[1]:,}, Non-Flood:{np.bincount(y_test.astype(int))[0]:,}')
# fig.savefig(rf"F:\CROP-PIER\CROP-WORK\Presentation\20210309\Fig\{strip_id}.png")
#%%
df_pred = pd.DataFrame(df.loc[(df["ext_act_id"]%10).isin([8, 9])])
df_pred = df_pred.drop(columns="label")
df_pred = df_pred.assign(pred_proba = model.predict_proba(x_test)[:, 1])
df_pred = df_pred.assign(label = y_test)
df_pred = df_pred.assign(bce_loss=(-df_pred["label"]*np.log(df_pred["pred_proba"]))-((1-df_pred["label"])*np.log(1-df_pred["pred_proba"])))
df_pred = df_pred.sort_values(by="bce_loss", ascending=False)
