import os
import sys
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import re
import datetime
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from skimage import filters
from general_fn import *
import matplotlib.pyplot as plt
tqdm.pandas()
#%% root 
root_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-old\Sentinel1_flood"
#%%
#observed_index = 'PLANT_PROVINCE_CODE'
report = True # Show hist
observed_index = 'strip_id'
index = [304] # Choose parameters from which strip_id 
strip_id = 304 # For which strip id
#%%  Load (only)flood dataframe
list_df = []
for file in os.listdir(root_flood):
    df_flood = pd.read_parquet(os.path.join(root_flood, file))
    list_df.append(df_flood)
df_flood = pd.concat(list_df, sort=False)
#print(df_flood['strip_id'].value_counts(sort=False))
#%% Start Process
windows_size = 2
t_column_index = np.where(df_flood.columns == 't')[0][0]
columns = df_flood.columns[t_column_index-windows_size:t_column_index+windows_size+1]
df_selected_p = df_flood[df_flood[observed_index].isin(index)]
#%% drop unnecessary data
df_selected_p = df_selected_p.dropna(subset=columns)
df_selected_p.drop(columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE'], inplace=True)
df_selected_p = df_selected_p[(df_selected_p['loss_ratio']>=0.8) & (df_selected_p['loss_ratio']<=1.0)]
df_selected_p = df_selected_p.reset_index(drop=True)
#%% 
# Add min(flood) and min(flood)_date columns
df_selected_p = df_selected_p.assign(min_water = df_selected_p[columns].min(axis=1),
                                     min_water_column = columns[np.argmin(df_selected_p[columns].values, axis=1)])

# Add change value of each date (compare to previous date) (old-new)
water_diff = -np.diff(df_selected_p[columns], axis=1)
diff_columns = np.array([f"change_{columns[i]}_{columns[i+1]}" for i in range(len(columns)-1)])
df_selected_p = df_selected_p.assign(**dict(zip(diff_columns, water_diff.T)))

# Add max(change value) and max(change value) columns
df_selected_p = df_selected_p.assign(max_change = df_selected_p[diff_columns].max(axis=1),
                                     max_change_column = diff_columns[np.argmax(df_selected_p[diff_columns].values, axis=1)],
                                     min_change = df_selected_p[diff_columns].min(axis=1),
                                     min_change_column = diff_columns[np.argmin(df_selected_p[diff_columns].values, axis=1)])
#%% Show distribution of flood date and max change date  
#print(df_selected_p['min_water_column'].value_counts())
#print()
#print(df_selected_p['max_change_column'].value_counts())
#print()
#print(df_selected_p['min_change_column'].value_counts())
#%% Show histograms of min water value, and max change column
#plt.figure()
#plt.hist(df_selected_p['min_water_column'], bins='auto')
#plt.figure()
#plt.hist(df_selected_p['max_change_column'], bins='auto')
#plt.figure()
#plt.hist(df_selected_p['min_change_column'], bins='auto')
#%% 
selected_percentile = 0.7
percentiles = [0.2, 0.4, 0.5, 0.6, 0.8]
#print(df_selected_p['min_water'].quantile(percentiles))
#print(df_selected_p['max_change'].quantile(percentiles))
#print(df_selected_p['min_change'].quantile(percentiles))


water_percentile = selected_percentile
change_to_water_percentile = 1-selected_percentile
change_from_water_percentile = selected_percentile

water_threshold =  df_selected_p['min_water'].quantile(selected_percentile)
change_to_water_threshold =  df_selected_p['max_change'].quantile(1-selected_percentile)
change_from_water_threshold =  df_selected_p['min_change'].quantile(selected_percentile)

#print()
print("water_thresh =", water_threshold)
print("to_water_thresh = ", change_to_water_threshold)
print("to_land_thresh = ", change_from_water_threshold)
print("water_thresh /w otsu:", filters.threshold_otsu(df_selected_p['min_water'][df_selected_p['min_water']<water_threshold].values, nbins=4096))
#%%
print(df_selected_p['strip_id'].value_counts())
print(df_selected_p['START_DATE'].value_counts())
#%% Report water pixel
if report:
    plt.close('all')
    plt.figure()
    plt.hist([df_selected_p['t'], df_selected_p['min_water']], bins='auto', label=['t', 'min(t-2:t+2)'])
    plt.xlim((0, 0.20))
    plt.legend(loc='upper right')
    for percentile_value, percentile in zip(df_selected_p['min_water'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='orange', linestyle='dashed', linewidth=1.5)
        plt.text(percentile_value, plt.ylim()[1], "P({0:.1f})".format(percentile), horizontalalignment = 'center')
        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.5f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Water pixel values histogram")
# Report max change value
if report:
    plt.figure()
    plt.hist(df_selected_p['max_change'], bins='auto')
    plt.xlim((-0.03, 0.16))
    for percentile_value, percentile in zip(df_selected_p['max_change'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(percentile_value, plt.ylim()[1]-40, "P({0:.1f})".format(percentile), horizontalalignment = 'center')
        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.4f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Max change values histogram")
# Report min change value
if report:
    plt.figure()
    plt.hist(df_selected_p['min_change'], bins='auto')
    plt.xlim((-0.16, 0.03))
    for percentile_value, percentile in zip(df_selected_p['min_change'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(percentile_value, plt.ylim()[1]-40, "P({0:.1f})".format(percentile), horizontalalignment = 'center')
        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.4f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Min change values histogram")






