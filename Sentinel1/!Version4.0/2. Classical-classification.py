import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
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
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_s1_temporal_2020 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"

root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
strip_id = "304"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
columns = df.columns[:31]

# Convert power to db
df = convert_power_to_db(df, columns)
#%%
x_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), columns].values
y_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), "label"].values
x_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), columns].values
y_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), "label"].values
#%%
# model = RandomForestClassifier(n_jobs=-1)
model = RandomForestClassifier(min_samples_leaf=5, max_depth=10, min_samples_split=10,
                               verbose=0, n_jobs=-1, random_state=42)
model.fit(x_train, y_train)
#%%
fig, ax = plt.subplots()
plot_roc_curve(model, x_train, y_train, label="train", color="b-", ax=ax)
plot_roc_curve(model, x_test, y_test, label="test", color="g-", ax=ax)
ax = set_roc_plot_template(ax)
#%% Load test data
df_2020 = pd.read_parquet(os.path.join(root_df_s1_temporal_2020, f"df_s1ab_pixel_s{strip_id}.parquet"))

# Drop mis-located polygon
df_2020 = df_2020.loc[df_2020["is_within"]]

# Drop tier 2
df_2020 = df_2020.loc[df_2020["tier"] == 1]

# df_2020.columns = [column[:8] if "_S1" in column else column for column in df_2020.columns]
columns = [column for column in df_2020.columns if "_S1" in column ]
list_p = df_2020["p_code"].unique().tolist()

# Load df vew
df_vew = pd.concat(
    [pd.read_parquet(os.path.join(root_df_vew_2020, file))
     for file in os.listdir(root_df_vew_2020)
     if file.split(".")[0].split("_")[-1] in list_p
     ],
    ignore_index=True
)
df_vew = df_vew.loc[df_vew["ext_act_id"].isin(df_2020["ext_act_id"].unique())]
df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
# Keep (non-flood and loss_ratio == 0 | flood and loss_ratio != 0)
df_vew = df_vew.loc[((df_vew["DANGER_TYPE_NAME"].isna()) & (df_vew["loss_ratio"] == 0)) | (~((df_vew["loss_ratio"] == 0) & (df_vew["DANGER_TYPE_NAME"] == 'อุทกภัย')))]

# Drop if at the edge
df_2020 = df_2020.loc[~(df_2020.iloc[:, 7:] == 0).any(axis=1)]

# Merge df_2020, df_vew
df_2020 = pd.merge(df_2020, df_vew[["ext_act_id", "final_plant_date", "PLANT_PROVINCE_CODE", "loss_ratio"]], on="ext_act_id", how="inner")

# Down sampling non-flood
df_2020 = df_2020[(df_2020["ext_act_id"].isin(np.random.choice(df_2020.loc[df_2020["loss_ratio"] == 0, "ext_act_id"].unique(), len(df_2020.loc[df_2020["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df_2020["loss_ratio"] >= 0.8)]

# Convert power to db
df_2020 = convert_power_to_db(df_2020, columns)

# Select only columns within crop cycle (each row)
list_arr = []
for _, row in tqdm(df_2020.iterrows(), total=len(df_2020)):
    final_plant_date = str(row["final_plant_date"].date()).replace("-", "")
    # Find plant_date column
    for idx, col in enumerate(columns):
        if col[:8] >= final_plant_date:
            break
    
    data = row[columns[idx:idx+31]].values
    if len(data) != 31:
        data = np.zeros((31, ))
    list_arr.append(data)

columns = df.columns[:31]
df_2020[columns] = np.vstack(list_arr)
print(df_2020.PLANT_PROVINCE_CODE.value_counts())
#%%
df_2020 = df_2020[df_2020["p_code"] == "30"]
fig, ax = plt.subplots()
df_2020.loc[df_2020["label"] == 0, columns].mean(axis=0).plot(ax=ax, label="normal_2020")
df_2020.loc[df_2020["label"] == 1, columns].mean(axis=0).plot(ax=ax, label="flood_2020")

df.loc[df["label"] == 0, columns].mean(axis=0).plot(ax=ax, label="normal_2018, 2019")
df.loc[df["label"] == 1, columns].mean(axis=0).plot(ax=ax, label="flood_2018, 2019")
plt.legend()
#%%
fig, ax = plt.subplots()
plot_roc_curve(model, df_2020[columns].values, df_2020["label"], label="2020", color="r-", ax=ax)
ax = set_roc_plot_template(ax)
#%%