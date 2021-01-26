import os
import datetime
import pandas as pd
import numpy as np
import icedumpy
from tqdm import tqdm

def load_ls8_pixel_dataframe_every_year(root, p, pathrow, band):  
    # Get all files
    list_file = []
    for file in os.listdir(root):
        if f"{p}_{pathrow}_{band}" in file:
            list_file.append(file)
    # sort year
    list_file.sort()
    
    # Load all files
    for file in list_file:
        if not "df" in locals():
            df = pd.read_parquet(os.path.join(root, file))
        else:
            df = pd.merge(df, pd.read_parquet(os.path.join(root, file)), how="inner", on=['new_polygon_id', 'row', 'col'])
    return df

def extract_pixel_value_in_period(df_vew, df_ls8, band):
    if not (df_ls8_pixel.index == df_ls8_pixel.new_polygon_id).all():
        df_ls8 = icedumpy.df_tools.set_index_for_loc(df_ls8, column = 'new_polygon_id')
    
    ls8_dates = pd.Series([datetime.datetime.strptime(date, "%Y-%m-%d") for date in df_ls8.columns[3:]])
    ls8_dates.index += 3
    
    list_df = []
    for start_date, df_vew_grp in df_vew.groupby(["START_DATE"]):
        # Check start_date in range or not
        if not (ls8_dates[ls8_dates.index[0]] <= start_date <= ls8_dates[ls8_dates.index[-1]]):
            continue
        
        # Find which column of landast8 dataframe corresponding to flood period
        start_date_column_index = ls8_dates[start_date<=ls8_dates].index[0]
        
        # Too early
        if (start_date_column_index-2)<3:
            continue
        
        window_date_column = df_ls8.columns[start_date_column_index-2:start_date_column_index+2]
        if len(window_date_column)!=4 :
            continue

        for i in range(len(df_vew_grp)):
            new_polygon_id = df_vew_grp.iloc[i, 0]
            
            # Get ls8 infomations of new_polygon_id
            df_ls8_selected = df_ls8.loc[new_polygon_id:new_polygon_id]
            
            # Get pixel values 
            polygon_pixel_values = df_ls8_selected[window_date_column].values
            
            # Get row, col
            rows = df_ls8_selected['row']
            cols = df_ls8_selected['col']
            
            # Skip if values out of image's frame (some column are "0")
            if (polygon_pixel_values==0).all(axis=0).any():
                continue
            
            if band=="BQA":
                if (polygon_pixel_values==1).all(axis=0).any():
                    continue    
            
            # Append data
            df = pd.DataFrame({'pathrow' : [pathrow]*len(polygon_pixel_values),
                               'row' : rows,
                               'col' : cols,
                               'new_polygon_id' : new_polygon_id,
                               'PLANT_PROVINCE_CODE' : df_vew_grp.iat[i, 1],
                               'PLANT_AMPHUR_CODE' : df_vew_grp.iat[i, 2],
                               'PLANT_TAMBON_CODE' : df_vew_grp.iat[i, 3],
                               'final_plant_date' : df_vew_grp.iat[i, 4],
                               'START_DATE' : start_date,
                               'loss_ratio' : df_vew_grp.iat[i, -1],
                             })
            for j in range(polygon_pixel_values.shape[1]):
                df[f"t{j-2:+d}"] = polygon_pixel_values[:, j]
            
            if len(df)!=0:
                list_df.append(df)
    if len(list_df)!=0:
        df = pd.concat(list_df, ignore_index=True)
        return df
    else:
        return pd.DataFrame()
#%%
root_df_ls8_pixel = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_pixel_from_mapping_v2"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_save_flood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_flood_value"
root_save_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_noflood_value"
os.makedirs(root_save_flood, exist_ok=True)
os.makedirs(root_save_noflood, exist_ok=True)
#%% Get valuable variables
# Total band from ls8_pixel
bands = np.unique([file.split("_")[4] for file in os.listdir(root_df_ls8_pixel)]).tolist()
years = np.unique([file.split("_")[5].split(".")[0] for file in os.listdir(root_df_ls8_pixel)]).tolist()

# Set of p, pathrow from vew
df_unique_p_pathrow = pd.DataFrame(np.unique(["_".join(file.split("_")[2:4]) for file in os.listdir(root_df_ls8_pixel)]).tolist(), columns=['p_pathrow'])
df_unique_p_pathrow['p'] = df_unique_p_pathrow['p_pathrow'].apply(lambda val: val.split("_")[0])
df_unique_p_pathrow['pathrow'] = df_unique_p_pathrow['p_pathrow'].apply(lambda val: val.split("_")[1])
#%%
#selected_pathrows = ["129048", "129049", "130048", "130049"]
#df_unique_p_pathrow = df_unique_p_pathrow[df_unique_p_pathrow['pathrow'].isin([selected_pathrows[0]])]
#df_unique_p_pathrow = df_unique_p_pathrow.sample(frac=1)
#%%
for p, df_unique_p_pathrow_selected_p in tqdm(df_unique_p_pathrow.groupby(['p'], sort=False)):
    df_vew = icedumpy.df_tools.load_vew(root_df_vew, [p[1:]])
    df_vew = icedumpy.df_tools.clean_and_process_vew(df_vew)
    df_vew = df_vew[["new_polygon_id", "PLANT_PROVINCE_CODE", "PLANT_AMPHUR_CODE", "PLANT_TAMBON_CODE", "final_plant_date", "START_DATE", "loss_ratio"]]
    
    for pathrow in df_unique_p_pathrow_selected_p['pathrow']:
        for idx, band in enumerate(bands):
            path_save_flood = os.path.join(root_save_flood, f"df_ls8_flood_pixel_{p}_{pathrow}_{band}.parquet")
            path_save_noflood = os.path.join(root_save_noflood, f"df_ls8_noflood_pixel_{p}_{pathrow}_{band}.parquet")
            if os.path.exists(path_save_flood) and os.path.exists(path_save_noflood):
                tqdm.write(f"FILE ALREADY EXISTS: {p}_{pathrow}_{band}")
                continue
            
            tqdm.write(f"START EXTRACTING: {p}_{pathrow}_{band}")
            df_ls8_pixel = load_ls8_pixel_dataframe_every_year(root_df_ls8_pixel, p, pathrow, band)
            df_ls8_pixel = icedumpy.df_tools.set_index_for_loc(df_ls8_pixel, column = 'new_polygon_id')
            if len(df_ls8_pixel)==0:
                pd.DataFrame().to_parquet(path_save_flood)
                pd.DataFrame().to_parquet(path_save_noflood)
                continue
            
            ls8_dates = pd.Series([datetime.datetime.strptime(date, "%Y-%m-%d") for date in df_ls8_pixel.columns[3:]])
            ls8_dates.index += 3
            
            if idx==0:
                # Get df_vew of the selected band and pathrow and filtered by df_ls8_pixel polygon_id
                df_vew_filtered_by_ls8_id = df_vew[df_vew['new_polygon_id'].isin(pd.unique(df_ls8_pixel['new_polygon_id']))]
    
                # vew flood dataframe
                df_vew_filtered_by_ls8_id_flood = df_vew_filtered_by_ls8_id[df_vew_filtered_by_ls8_id['loss_ratio']!=0]
                
                # vew no flood dataframe
                df_vew_filtered_by_ls8_id_noflood = df_vew_filtered_by_ls8_id[df_vew_filtered_by_ls8_id['loss_ratio']==0]
                # Randomly add 1-4 months date into final_plant_date then fill in start_date (not more than the last raster date)
                df_vew_filtered_by_ls8_id_noflood = df_vew_filtered_by_ls8_id_noflood[((ls8_dates.iloc[-1] - df_vew_filtered_by_ls8_id_noflood['final_plant_date']).dt.days) >= 30]
                timedelta = pd.to_timedelta([np.random.randint(30, min(121, (ls8_dates.iloc[-1]-date).days+1)) for date in df_vew_filtered_by_ls8_id_noflood['final_plant_date']], unit='days')
                df_vew_filtered_by_ls8_id_noflood['START_DATE'] = df_vew_filtered_by_ls8_id_noflood['final_plant_date']+timedelta          
                
                del df_vew_filtered_by_ls8_id
            
            # Extract flood
#            df_result = extract_pixel_value_in_period(df_vew_filtered_by_ls8_id_flood, df_ls8_pixel, band) 
#            df_result.to_parquet(path_save_flood)
            
            # Extract no flood
            df_result = extract_pixel_value_in_period(df_vew_filtered_by_ls8_id_noflood, df_ls8_pixel, band) 
            df_result.to_parquet(path_save_noflood)

#%%