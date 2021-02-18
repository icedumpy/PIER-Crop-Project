import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
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

def load_df(root_df_s1_temporal, strip_id):
    columns = [f"t{i}" for i in range(31)]
    list_df = []
    for file in tqdm(os.listdir(root_df_s1_temporal)):
        if file.split(".")[0][-3:] == strip_id:
            df = pd.read_parquet(os.path.join(root_df_s1_temporal, file))
            df = df.drop(columns=["row", "col"])
            # Replace negatives with nan
            for col in columns:
                df.loc[df[col] <= 0, col] = np.nan
            
            # Drop row with mostly nan
            df = df.loc[df[columns].isna().sum(axis=1) < 10]
            
            # Interpolate nan values
            df[columns] = interp_numba(df[columns].values)
            
            # Convert power to dB 
            df[columns] = 10*np.log10(df[columns])    
            
            # Agg data (mean and std)
            df_temp = df[columns+["ext_act_id", "loss_ratio"]].groupby(["ext_act_id", "loss_ratio"]).agg(["mean", "std"])
            
            # Rename column: mean: tx >> mx, std tx >> sx
            df_temp.columns = [col[0].replace("t", "m") if col[1] == "mean" else col[0].replace("t", "s") for col in df_temp.columns.values]
            
            # If std is nan >> Replace with 0
            df_temp[df_temp.columns[df_temp.columns.str.contains("s")]] =  df_temp[df_temp.columns[df_temp.columns.str.contains("s")]].fillna(0)
            
            # Re-ordering column
            df_temp = df_temp.reindex(sorted(df_temp.columns), axis=1)
            
            # Reset index (Move ext_act_id and loss_ratio into columns)
            df_temp = df_temp.reset_index() 
            
            # Merge agg with some remain info
            df = df[[column for column in df.columns if not column in columns]].drop_duplicates()
            df = pd.merge(df_temp, df[[column for column in df.columns if not column in columns]], how="inner")
            
            # Calculate start_date (find tx) if no flood >> t(-1) else t(x)
            df.loc[df["loss_ratio"] > 0, "START_DATE"] = ((df.loc[df["loss_ratio"] > 0, "START_DATE"] - df.loc[df["loss_ratio"] > 0, "final_plant_date"]).dt.days//6)
            df.loc[df["loss_ratio"] == 0, "START_DATE"] = -1
            df["START_DATE"] = df["START_DATE"].astype("int")
            df = df.loc[(df["loss_ratio"] == 0).sample(frac=0.1) | (df["loss_ratio"] != 0)]
            
            list_df.append(df)
    df = pd.concat(list_df, ignore_index=True)
    return df

def change_date(column, const):
    """
    Add or subtract column_t ex. ("t20", 5) >> "t25"
    Examples
    --------
    >>> change_t("t20", 5) >> "t25"
    >>> change_t("s20", -2) >> "s18"
    """
    return f"{column[0]}{int(column[1:])+const}"

def get_columns_range(column, char, window):
    return [f"{char}{i}" for i in range(int(column[1:])+window[0], int(column[1:])+window[1]+1)]
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
strip_id = "402"
# [(column, 6*int(column[1:])) for column in columns]
columns_mean = [f"m{i}" for i in range(31)]
columns_std  = [f"s{i}" for i in range(31)]
age_group1 = [f"t{i}" for i in range(0, 7)]
age_group2 = [f"t{i}" for i in range(6, 15)]
age_group3 = [f"t{i}" for i in range(14, 20)]
age_group4 = [f"t{i}" for i in range(19, 31)]
#%%
# Load data
df = load_df(root_df_s1_temporal, strip_id)
#%%
df_flood = df[df["START_DATE"] != -1]
#%%
list_mean = []
list_std = []
for index, row in df_flood.iterrows():
    if row.START_DATE >= 3 and row.START_DATE <= 28:
        list_mean.append(row[get_columns_range(f"m{row.START_DATE}", "m", (-2, 2))].values)
        list_std.append(row[get_columns_range(f"s{row.START_DATE}", "s", (-2, 2))].values)
mean = np.vstack(list_mean)
std = np.vstack(list_std)
#%%
plt.plot(mean.mean(axis=0))
#%%
#%%
list_diff_mean_f = []
list_diff_mean_nf = []
# for index, row in tqdm(df[df["loss_ratio"] > 0].iterrows(), total=len(df[df["loss_ratio"] > 0])):
for index, row in tqdm(df.iterrows(), total=len(df)):
    if row.loss_ratio > 0 :
        if row.START_DATE >= 3 and row.START_DATE <= 28:
            column = f"m{row['START_DATE']}"
            diff = (row[get_columns_range(column, "m", (0, 2))].min() - row[get_columns_range(column, "m", (-3, -1))].mean())
            # diff = np.diff(row[get_columns_range(column_flood, "m", (3, 2))])
            list_diff_mean_f.append(diff.mean())
    else:
        column = f"m{np.random.randint(3, 29)}"
        diff = (row[get_columns_range(column, "m", (0, 2))].min() - row[get_columns_range(column, "m", (-3, -1))].mean())
        # diff = np.diff(row[get_columns_range(column, "m", (3, 2))])
        list_diff_mean_nf.append(diff.mean())
#%%
sns.histplot(list_diff_mean_f, label="Change(flood)")
sns.histplot(list_diff_mean_nf, label="Change(non-flood)")
plt.legend()
plt.title("Min(t, t+1, t+2)-Mean(t-3, t-2, t-1)")
#%%



#%%
row = df.loc[df["loss_ratio"] > 0].sample(n=1).squeeze()
#%%
plt.close('all')
column_flood = f"m{row['START_DATE']}"
plt.plot(row[columns_mean])
plt.axvline(f"{column_flood}", color="r", linestyle="-.")
plt.grid()
plt.title(f"{row['loss_ratio']}")
#%% -3, +2 seems to be okay
# Get back scattering
dn = row[change_date(column_flood, -3):change_date(column_flood, +2)]
#%%
# Check downward trend (After flood - mean(before_flood)) Should be at least one period of high negative
downtrend = (row[change_date(column_flood, 0):change_date(column_flood, 2)] - row[change_date(column_flood, -3):change_date(column_flood, -1)].mean()).values
#%%
# Find cumulative change
diff = np.diff(row[change_date(column_flood, -3):change_date(column_flood, +2)])
diff_cumsum = diff.cumsum()[1:] # Exclude first element ()
#%%
plt.plot(np.hstack([dn, downtrend, diff, diff_cumsum]))
#%%
diff.mean()

#%%
#%%
# #%% Case Flood
# plt.close('all')s
# try:
#     df_sample = df.loc[df["loss_ratio"] >= 0.8].sample(n=1)
#     df_sample[columns] = df_sample[columns].interpolate(limit_direction="both", axis=1)
#     print(f"t{df_sample['START_DATE'].values.astype(int)[0]}")
#     arr = df_sample[columns].values[0]
#     plt.plot(columns, arr, "--")
#     plt.axvline(f"t{df_sample['START_DATE'].values.astype(int)[0]}", color="r", linestyle="-.")
#     # plt.plot(columns, arr - df_mean_normal_province.loc[df_sample["PLANT_PROVINCE_CODE"]].values[0])
#     plt.plot(columns, df_mean_normal_province.loc[df_sample["PLANT_PROVINCE_CODE"]].values[0])
#     # plt.plot(np.ediff1d(arr, to_begin=0))
#     plt.grid()
#     print(arr.std())
# except:
#     print(arr)
# #%% Case normal
# plt.close('all')
# try:
#     df_sample = df.loc[df["loss_ratio"] == 0.0].sample(n=1)
#     df_sample[columns] = df_sample[columns].interpolate(limit_direction="both", axis=1)
#     arr = df_sample[columns].values[0]
#     plt.plot(columns, arr, "--")
#     plt.plot(columns, df_mean_normal_province.loc[df_sample["PLANT_PROVINCE_CODE"]].values[0])
#     # plt.plot(np.ediff1d(arr, to_begin=0))
#     plt.grid()
#     print(arr.std())
# except:
#     print(arr)
#%%







