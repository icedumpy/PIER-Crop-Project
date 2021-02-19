import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt
from supervised.automl import AutoML
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
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
strip_id = "402"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
df = df[(df["ext_act_id"].isin(np.random.choice(df.loc[df["loss_ratio"] == 0, "ext_act_id"].unique(), len(df.loc[df["loss_ratio"] >= 0.8, "ext_act_id"].unique()), replace=False))) | (df["loss_ratio"] >= 0.8)]
columns = df.columns[:31]
#%%           
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
#%%
x_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), columns].values
y_train = df.loc[~(df["ext_act_id"]%10).isin([8, 9]), "label"].values
x_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), columns].values
y_test = df.loc[(df["ext_act_id"]%10).isin([8, 9]), "label"].values
#%%
automl = AutoML()
automl.fit(x_train, y_train)
predictions = automl.predict(x_test)

