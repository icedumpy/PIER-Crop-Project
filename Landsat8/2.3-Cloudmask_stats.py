import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import icedumpy

def get_file_dataframe(root, pathrows, bands):
    df = pd.DataFrame(columns=['p', 'pathrow', 'band', 'path_file'])
    for file in os.listdir(root):
        p = file.split("_")[4]
        pathrow = file.split("_")[5]
        band = file.split(".")[0].split("_")[6]

        if (pathrow in pathrows) and (band in bands):
            df = df.append(pd.Series({'p' : p,
                                      'pathrow' : pathrow,
                                      'band' : band,
                                      'path_file' : os.path.join(root, file)}),
                           ignore_index=True)
    return df

def merge_and_concat_file_dataframe(df_file, ratio=1):
    list_df = []
    for (p, pathrow), df_file_grp in tqdm(df_file.groupby(['p', 'pathrow'])):
        for index, row in df_file_grp.iterrows():
            df_temp = pd.read_parquet(row['path_file'])
            if len(df_temp)==0:
                continue
            
            band = row['band']
            df_temp.columns = df_temp.columns[:-4].tolist() + [f"{band}({item})" for item in df_temp.columns[-4:]]
            
            if not "df_merge_band" in locals():
                df_merge_band = df_temp
            else:
                df_merge_band = pd.merge(df_merge_band, df_temp, how='inner', on=df_temp.columns[:10].tolist())
        
        # After finishing load and merge all of the selected bands, drop duplicates and append df_merge_band to list_df
        if "df_merge_band" in locals():
            if df_merge_band.shape[1]==(10+len(bands)*4):
                # equivalent to subset = df_merge_band.columns[10:] but a lot faster
                df_merge_band = df_merge_band.drop_duplicates(subset=['row', 'col', 'START_DATE'])
                # Sampling down a bit :( (Needed, especially for no flood)
                if ratio!=1:
                    df_merge_band = df_merge_band.sample(frac=ratio)
                list_df.append(df_merge_band)
                del df_merge_band
                
    df = pd.concat(list_df, ignore_index=True)
    return df

def add_cloudmask(df, df_cloudmask):
    cloudmask_date = pd.Series([datetime.datetime.strptime(item, "%Y-%m-%d") for item in df_cloudmask.columns[3:]])
    cloudmask_date.index += 3
    list_df = []
    for start_date, df_grp in tqdm(df.groupby(['START_DATE'])):
        pass
        # Get (t-2, t-1, t+0, t+1) columns from cloudmask
        if not (cloudmask_date[cloudmask_date.index[0]] <= start_date <= cloudmask_date[cloudmask_date.index[-1]]):
            continue
        start_date_column_index = cloudmask_date[start_date<=cloudmask_date].index[0]
        window_date_column = df_cloudmask.columns[start_date_column_index-2:start_date_column_index+2]
    
        # Get cloudmask pixel data
        rows_cols = df_grp['row'].astype(str) + "_" + df_grp['col'].astype(str)
        polygon_pixel_cloudmask = np.zeros((len(rows_cols), 4), dtype='uint8')
        for idx, row_col in enumerate(rows_cols):
            try:
                polygon_pixel_cloudmask[idx] = df_cloudmask.loc[row_col][window_date_column].values
            except:
                polygon_pixel_cloudmask[idx] = np.zeros((1, 4), dtype='uint8')
    
        # Add cloudmask pixel data in to original df
        for j in range(polygon_pixel_cloudmask.shape[1]):
            df_grp.insert(loc=len(df_grp.columns), column=f"cloudmask_t{j-2:+d}", value=polygon_pixel_cloudmask[:, j])
        list_df.append(df_grp)
    
    df = pd.concat(list_df, ignore_index=True)
    return df

def get_stats(df):
    column_sum = df[df.columns[-4:]].values.sum(axis=1)
    _, stats = np.unique(column_sum, return_counts=True)
    stats = pd.Series(stats)
    df_stats = pd.DataFrame(stats, columns = ["Cloud counts"])
    df_stats['Percentage'] = 100*df_stats['Cloud counts']/df_stats['Cloud counts'].sum()
    return df_stats

def main(root_df, root_cloudmask, pathrows, bands):
    # Get dataframe of file path of selecteh pathrow and band
    df_file = get_file_dataframe(root_df, pathrows, bands)
    
    # Load data from dataframe of file path
    df = merge_and_concat_file_dataframe(df_file)
    df = df.sample(n=min(200000, len(df)))
    
    filter_row_col = df['row'].astype(str) + "_" + df['col'].astype(str)
    df.insert(loc=3, column='row_col', value=filter_row_col)
    
    # Load cloudmask
    df_cloudmask = icedumpy.df_tools.load_ls8_cloudmask_dataframe(root_cloudmask, pathrows[0], filter_row_col=filter_row_col)
    df_cloudmask = icedumpy.df_tools.set_index_for_loc(df_cloudmask, index=df_cloudmask['row'].astype(str) + "_" + df_cloudmask['col'].astype(str))
    
    df = add_cloudmask(df, df_cloudmask)
    return df
#%%
root_flood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_flood_value_new"
root_cloudmask = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_cloudmask"
root_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_noflood_value_new"

pathrows = ['130048', '130049', '129048', '129049']
pathrows = ['129049']
bands = ['B2', 'B3', 'B4', 'B5']
ratio_for_noflood = 0.3

root_save = rf"C:\Users\PongporC\Desktop\9-7-2020\Stats\{pathrows[0]}"
os.makedirs(root_save, exist_ok=True)
#%%
df_flood = main(root_flood, root_cloudmask, pathrows, bands)
df_noflood = main(root_noflood, root_cloudmask, pathrows, bands)
#%%
df_flood_cloud_stats = get_stats(df_flood)
df_noflood_cloud_stats = get_stats(df_noflood)
df_flood_stats = pd.concat((df_flood.iloc[:, 11:-4].mean().rename("Mean"), df_flood.iloc[:, 11:-4].std().rename("Std")), axis=1)
df_noflood_stats = pd.concat((df_noflood.iloc[:, 11:-4].mean().rename("Mean"), df_noflood.iloc[:, 11:-4].std().rename("Std")), axis=1)
#%%
df_flood_stats.to_csv(os.path.join(root_save, "df_flood_stats.csv"))
df_flood_cloud_stats.to_csv(os.path.join(root_save, "df_flood_cloud_stats.csv"))

df_noflood_stats.to_csv(os.path.join(root_save, "df_noflood_stats.csv"))
df_noflood_cloud_stats.to_csv(os.path.join(root_save, "df_noflood_cloud_stats.csv"))



