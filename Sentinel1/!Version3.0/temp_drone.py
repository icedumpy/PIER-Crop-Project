import os
import pandas as pd
#%%
path_df_drone = r"F:\CROP-PIER\CROP-WORK\drone-area-ext_act_id-2020.csv"
path_df_prediction = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020_Drop_FA\s1_prediction_p30_s304.parquet"
#%%
df_drone = pd.read_csv(path_df_drone)
df_prediction = pd.read_parquet(path_df_prediction)