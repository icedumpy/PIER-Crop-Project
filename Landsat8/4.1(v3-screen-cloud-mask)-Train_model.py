import os
import datetime
import numpy as np 
import pandas as pd
import icedumpy
#%%
root_cloudmask = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_cloudmask"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_flood_value"
root_df_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_noflood_value"
#%%
pathrows = ['129048', '129049', '130048', '130049']
bands = ['B2', 'B3', 'B4', 'B5']
#%%
for pathrow in pathrows:
    df_cloudmask = icedumpy.df_tools.load_ls8_cloudmask_dataframe(root_cloudmask, pathrow, filter_row_col=None)
    
    # Change date columns to datetime format
    dates_cloudmask = pd.Series([datetime.datetime.strptime(date.split("_")[-1], "%Y%m%d") for date in df_cloudmask.columns[3:]])
    dates_cloudmask.index += 3
    break
#%%
ice = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_flood_value\df_ls8_flood_pixel_p30_129049_B1.parquet")
start_date = ice.loc[0, 'START_DATE'] 
#%%
for start_date, _ in ice.groupby(['START_DATE']):
    start_date_column_index = dates_cloudmask[start_date<=dates_cloudmask].index[0] 
    print(start_date, df_cloudmask.columns[start_date_column_index])
    
    df_cloudmask.iloc[:, start_date_column_index-2:start_date_column_index+2].sum()
