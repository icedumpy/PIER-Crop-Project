import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from icedumpy.df_tools import set_index_for_loc
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

    return df
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
#%%
strip_id = "304"
#%%
# Load df_s1_temporal
df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_s{strip_id}.parquet"))
df_s1_temporal = set_index_for_loc(df_s1_temporal, column="ext_act_id")

# Drop mis-located polygon
df_s1_temporal = df_s1_temporal.loc[df_s1_temporal["is_within"]]

# Drop tier 2
df_s1_temporal = df_s1_temporal.loc[df_s1_temporal["tier"] == 1]

df_s1_temporal.columns = [column[:8] if "_S1" in column else column for column in df_s1_temporal.columns]
list_p = df_s1_temporal["p_code"].unique().tolist()

# Load df vew
df_vew = pd.concat(
    [pd.read_parquet(os.path.join(root_df_vew, file))
     for file in os.listdir(root_df_vew)
     if file.split(".")[0].split("_")[-1] in list_p
     ],
    ignore_index=True
)
df_vew = df_vew.loc[df_vew["ext_act_id"].isin(df_s1_temporal["ext_act_id"].unique())]
df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
# Keep (non-flood and loss_ratio == 0 | flood and loss_ratio != 0)
df_vew = df_vew.loc[((df_vew["DANGER_TYPE_NAME"].isna()) & (df_vew["loss_ratio"] == 0)) | (~((df_vew["loss_ratio"] == 0) & (df_vew["DANGER_TYPE_NAME"] == 'อุทกภัย')))]

# Drop if at the edge
df_s1_temporal = df_s1_temporal.loc[~(df_s1_temporal.iloc[:, 7:] == 0).any(axis=1)]

# Create df look up table (for df_s1_temporal column)
df_content = pd.DataFrame(
    [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(9, len(df_s1_temporal.columns)-1)],
    columns=["index", "start", "stop"]
)
df_content = df_content.set_index("index")
df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
df_content = df_content.sort_index()
#%%
# Down sampling non-flood
df_flood = df_vew.loc[df_vew["loss_ratio"] >= 0.8]
df_nonflood = df_vew.loc[df_vew["loss_ratio"] == 0]
df_nonflood["START_DATE"] = df_nonflood["final_plant_date"]

df_vew = pd.concat([df_flood, df_nonflood.sample(n=len(df_flood))])
df_vew["START_DATE"] = pd.to_datetime(df_vew["START_DATE"])

# Convert power to dB
df_s1_temporal =  convert_power_to_db(df_s1_temporal, df_s1_temporal.columns[7:])

df_vew = df_vew.sample(frac=1)
#%%
#%%
ylim=(-20, 0)
count_flood = 1
count_nonflood = 1
for row in tqdm(df_vew.itertuples(), total=len(df_vew)):
    try:
        loss_ratio = row.loss_ratio
        ext_act_id = row.ext_act_id
        date_plant = row.final_plant_date
        date_flood = row.START_DATE
        polygon_area_in_square_m = df_s1_temporal.loc[ext_act_id, "polygon_area_in_square_m"].iloc[0]
        column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
        column_flood = df_content.loc[((df_content["start"] < date_flood) & (df_content["stop"] >= date_flood))].index[0]
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(*ylim))
        # Draw group age
        ax.axvspan(0.0, 6.5, alpha=0.2, color='red')
        ax.axvspan(6.5, 15.0, alpha=0.2, color='green')
        ax.axvspan(15.0, 20, alpha=0.2, color='yellow')
        ax.axvspan(20.0, 29, alpha=0.2, color='purple')
        
        # Add group descriptions
        ax.text(2.5, ylim[-1]+0.25, "0-40 days")
        ax.text(10, ylim[-1]+0.25, "40-90 days")
        ax.text(17.5, ylim[-1]+0.25, "90-120 days")
        ax.text(22.5, ylim[-1]+0.25, "120+ days")
    
        # Get x, y
        # y = df_s1_temporal.loc[ext_act_id, df_s1_temporal.columns[column_plant:column_plant+30]].mean(axis=0).values
        y = df_s1_temporal.loc[ext_act_id, df_s1_temporal.columns[column_plant:column_plant+30]].T
        x = df_s1_temporal.columns[column_plant:column_plant+30].values
        
        # Plot x, y
        # ax.plot(x, y, linestyle="--", marker="o", color="blue", label=label)
        ax.plot(x, y)
        
        # Draw flood date line
        if loss_ratio >= 0.8:
            ax.axvline(df_s1_temporal.columns[column_flood], linestyle="--", color="red")
            ax.text(df_s1_temporal.columns[column_flood], ax.get_ylim()[1]+0.65, "Reported flood date", horizontalalignment="center", color="red")\
        
        # Add details
        ax.grid()        
        # ax.legend(loc=4)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{i}" for i in x], rotation="65")
        fig.suptitle(f"S:{strip_id}, P:{row.PLANT_PROVINCE_CODE}, EXT_ACT_ID:{int(ext_act_id)}\nPolygon area:{polygon_area_in_square_m:.2f} (m\N{SUPERSCRIPT TWO})\n Loss ratio:{loss_ratio:.2f}")
        
        # Savefig
        if loss_ratio >= 0.8:
            fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210311\Fig\flood", f"{count_flood}.png"), bbox_inches="tight")
            count_flood+=1
        else:
            fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210311\Fig\nonflood", f"{count_nonflood}.png"), bbox_inches="tight")
            count_nonflood+=1
    except:
        print(row)
#%%
# plt.rcParams["font.family"] = "tahoma"
# fig, ax = plt.subplots(figsize=(16, 9))
# sns.histplot((df_vew["START_DATE"]-df_vew["final_plant_date"]).dt.days.values, kde=True, ax=ax, bins=90)
# fig.suptitle(f"Total polygon(loss ratio >= 80%): {len(df_vew)}, S:{strip_id}, Year:2020")
# ax.set_xlabel("อายุข้าวตอนท่วม")
# ax.axvspan(0,   40 , alpha=0.2, color='red')
# ax.axvspan(40,  90 , alpha=0.2, color='green')
# ax.axvspan(90,  120, alpha=0.2, color='yellow')
# ax.axvspan(120, 180, alpha=0.2, color='purple')
# # Add group descriptions
# ylim = ax.get_ylim()
# ax.text(20, ylim[-1]+2, "0-40 days", horizontalalignment="center")
# ax.text(65, ylim[-1]+2, "40-90 days", horizontalalignment="center")
# ax.text(105, ylim[-1]+2, "90-120 days", horizontalalignment="center")
# ax.text(150, ylim[-1]+2, "120+ days", horizontalalignment="center")
# plt.savefig(rf"F:\CROP-PIER\CROP-WORK\Presentation\20210310\อายุข้าวตอนท่วม(S{strip_id}-Y2020).png", bbox_inches="tight")



