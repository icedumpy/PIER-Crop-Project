import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

percentiles = [0.2, 0.4, 0.5, 0.6, 0.8]
def plot_percentile_hist(index):   
    print(f"Start : {index}")
    df_flood = pd.concat([pd.read_parquet(os.path.join(root_flood, file)) for file in os.listdir(root_flood) if index in file], ignore_index=True)
    
    t_column_index = np.where(df_flood.columns == 't')[0][0]
    columns = df_flood.columns[t_column_index-2:t_column_index+2+1]

    df_flood = df_flood[~np.all(df_flood[columns]==0, axis=1)]
    df_flood = df_flood[~np.any(pd.isna(df_flood[columns]), axis=1)]
    
    df_flood = df_flood.dropna(subset=columns)
    df_flood.drop(columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE'], inplace=True)
    df_flood = df_flood[(df_flood['loss_ratio']>=0.8) & (df_flood['loss_ratio']<=1.0)]
    df_flood = df_flood.reset_index(drop=True)
    print(f"Total row: {len(df_flood)}")
    
    # Add min(flood) and min(flood)_date columns
    df_flood = df_flood.assign(min_water = df_flood[columns].min(axis=1),
                                         min_water_column = columns[np.argmin(df_flood[columns].values, axis=1)])
    
    # Add change value of each date (compare to previous date) (old-new)
    water_diff = -np.diff(df_flood[columns], axis=1)
    diff_columns = np.array([f"change_{columns[i]}_{columns[i+1]}" for i in range(len(columns)-1)])
    df_flood = df_flood.assign(**dict(zip(diff_columns, water_diff.T)))
    
    # Add max(change value) and max(change value) columns
    df_flood = df_flood.assign(max_change = df_flood[diff_columns].max(axis=1),
                               max_change_column = diff_columns[np.argmax(df_flood[diff_columns].values, axis=1)],
                               min_change = df_flood[diff_columns].min(axis=1),
                               min_change_column = diff_columns[np.argmin(df_flood[diff_columns].values, axis=1)])
    
    plt.close('all')
    plt.figure()
#    plt.hist([df_flood['t'], df_flood['min_water']], bins='auto', label=['t', 'min(t-2:t+2)'])
    plt.hist([df_flood['min_water']], bins='auto')
    plt.xlim((0, 0.20))
    plt.legend(loc='upper right')
    for percentile_value, percentile in zip(df_flood['min_water'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='orange', linestyle='dashed', linewidth=1.5)
        plt.text(percentile_value, plt.ylim()[1], "P({0:.1f})".format(percentile), horizontalalignment = 'center')
#        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.5f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Water pixel values histogram")

    plt.figure()
    plt.hist(df_flood['max_change'], bins='auto')
    plt.xlim((-0.03, 0.16))
    for percentile_value, percentile in zip(df_flood['max_change'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(percentile_value, plt.ylim()[1]-40, "P({0:.1f})".format(percentile), horizontalalignment = 'center')
#        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.4f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Max change values histogram")
   
    plt.figure()
    plt.hist(df_flood['min_change'], bins='auto')
    plt.xlim((-0.16, 0.03))
    for percentile_value, percentile in zip(df_flood['min_change'].quantile(percentiles), percentiles):
        plt.axvline(x=percentile_value, color='r', linestyle='dashed', linewidth=1)
        plt.text(percentile_value, plt.ylim()[1]-40, "P({0:.1f})".format(percentile), horizontalalignment = 'center')
#        plt.text(percentile_value, plt.ylim()[1]-80, "{0:.4f}".format(percentile_value), horizontalalignment = 'center')
    plt.title("Min change values histogram")    
    return df_flood
#%%
root_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_flood_pixel"
#%%
#for p_code in np.unique([os.path.splitext(file)[0][-8:-5] for file in os.listdir(root_flood)]):
index = 's304'
#index = 'p45'
df_flood = plot_percentile_hist(index)