import os
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
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

def read_and_flatten_temporal(list_path):
    list_df = []
    for path in tqdm(list_path):
        df = pd.read_parquet(path)
        df = df[df["in_season_rice_f"] == 1]
        df = df[(df["DANGER_TYPE"] == "อุทกภัย") | (df["DANGER_TYPE"]).isna()]
        df = df[(df["loss_ratio"] >= 0) & (df["loss_ratio"] <= 1)]
        df = convert_power_to_db(df, columns[:30])
        df = df.drop_duplicates(subset=columns[:30])
        df.loc[df["DANGER_TYPE_GRP"].isna(), "DANGER_TYPE_GRP"] = "Normal"

        # Flatten
        df_normal = df[df["DANGER_TYPE_GRP"] == "Normal"]
        df_flood  = df[df["DANGER_TYPE_GRP"] == "Flood"]
        df = pd.concat([
            pd.DataFrame({
                "Backscatter":df_normal[columns[:30]].values.reshape(-1),
                "Danger_type":["Normal"]*(len(df_normal)*30)
            }),
            pd.DataFrame({
                "Backscatter":df_flood[columns[:30]].values.reshape(-1),
                "Danger_type":["Flood"]*(len(df_flood)*30)
            })
        ], ignore_index=True) 
        list_df.append(df)
    df = pd.concat(list_df, ignore_index=True)
    return df
#%%
root_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211118"
columns = [
    't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 
    't8', 't9', 't10','t11', 't12', 't13', 't14', 't15', 
    't16', 't17', 't18', 't19', 't20', 't21', 't22', 
    't23', 't24', 't25', 't26', 't27', 't28', 't29',
    'DANGER_TYPE_GRP', 'loss_ratio'
]
#%% Get info
list_info = []
for file in os.listdir(root_temporal):
    list_info.append({
        "filename": file,
        "path": os.path.join(root_temporal, file),
        "p_code": file[-15:-13],
        "strip_id": file[-11:-8]
    })
df_info = pd.DataFrame(list_info)
#%%
df = read_and_flatten_temporal(df_info["path"])
df = pd.concat([df[df["Danger_type"] == "Normal"].sample(n=len(df[df["Danger_type"] == "Flood"])), df[df["Danger_type"] == "Flood"]], ignore_index=True)
fig, ax = plt.subplots()
sns.histplot(
    data=df, x="Backscatter", hue="Danger_type", ax=ax,
    common_norm=True, stat="probability"
)
ax.set_xlim(-30, 0)
ax.set_title("All scenes")
fig.savefig(os.path.join(root_save, "All_scenes.png"), bbox_inches="tight")

for i, strip_id in enumerate(sorted(df_info.strip_id.unique()), start=1):
    plt.close("all")
    print(strip_id)
    df = read_and_flatten_temporal(df_info.loc[df_info["strip_id"] == strip_id, "path"])
    df = pd.concat([df[df["Danger_type"] == "Normal"].sample(n=len(df[df["Danger_type"] == "Flood"])), df[df["Danger_type"] == "Flood"]], ignore_index=True)
    fig, ax = plt.subplots()
    sns.histplot(
        data=df, x="Backscatter", hue="Danger_type", ax=ax,
        common_norm=True, stat="probability"
    )
    ax.set_xlim(-30, 0)
    ax.set_title(f"Scene: {strip_id}")
    fig.savefig(os.path.join(root_save, f"{strip_id}.png"), bbox_inches="tight")
#%%