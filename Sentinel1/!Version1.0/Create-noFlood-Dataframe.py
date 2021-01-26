import os 
import sys
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import datetime
import pandas as pd
import rasterio 
import numpy as np
from tqdm import tqdm
#%%
root_rowcol_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV"

root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_noflood_pixel_from_mapping_v3"
#%%
# Loop by vew's dataframe
for filename in os.listdir(root_vew):
    
    # Get p from filename : Ex. p = 'p10'
    p = os.path.splitext(filename)[0][-3:]
    
    # Load vew's dataframe
    df_vew = pd.read_parquet(os.path.join(root_vew, filename))
    
    # Select only vew with no flood (no disasters)
    df_vew = df_vew[pd.isnull(df_vew['DANGER_TYPE_NAME'])]
    df_vew = df_vew[pd.isna(df_vew['DANGER_TOTAL_WA'])]
    
    # Add loss_ratio=0 column
    df_vew = df_vew.assign(loss_ratio = 0)
    
    df_vew['final_plant_date'] = pd.to_datetime(df_vew['final_plant_date'])

    # =============================================================================
    # Load all of vew's mapping dataframe (all mapping dataframe with 'pxx' in name) then loop for each mapping
    # =============================================================================
    list_path_mapping = [path_mapping for path_mapping in os.listdir(root_rowcol_mapping) if p in path_mapping]        
    for path_mapping in list_path_mapping:
        
        # Get strip_id from mapping filename
        strip_id = int(os.path.splitext(path_mapping)[0][-3:])
        
        # If save file already exist, skip to next mapping
        path_save = os.path.join(root_save, f"df_s1_noflood_pixel_{p}_s{strip_id}.parquet")
        if os.path.exists(path_save):
            tqdm.write(f"Skip: {path_save}")
            continue
        
        tqdm.write(f"START: {path_save}")
        
        # Load mapping dataframe
        df_mapping = pd.read_parquet(os.path.join(root_rowcol_mapping, path_mapping))
        
        # Select only vew that vew's new_polygon_id is in mapping's new_polygon_id
        df_vew_grp = df_vew[df_vew['new_polygon_id'].isin(df_mapping.index)]
        
        # Skip mapping if no matched new_polygon_id
        if len(df_vew_grp)==0:
            # Write something
            tqdm.write(f"Skip: {path_mapping} (No dataframe in mapping)")
            continue
        
        # Load raster, raster_date, raster_im
        raster = rasterio.open(os.path.join(root_raster, f"LSVVS{strip_id}_2017-2019"))
        raster_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
        raster_im = raster.read()

        # Create ideal raster date and actual raster date dataframe (for flood pixels mapping)
        df_date = pd.DataFrame(pd.date_range(raster_date[0], raster_date[-1], freq='12D'), columns=['ideal'])
        df_date['available'] = df_date['ideal'].isin(raster_date)
        df_date['raster_band'] = (df_date['available'].cumsum()).astype('int') - 1
        df_date.loc[~df_date['available'], 'raster_band'] = 0
        
        # Randomly add 1-4 months date into final_plant_date then fill in start_date (not more than the last raster date)
        df_vew_grp = df_vew_grp[((raster_date[-1] - df_vew_grp['final_plant_date']).dt.days) >= 30]
        timedelta = pd.to_timedelta([np.random.randint(30, min(121, (raster_date[-1]-date).days+1)) for date in df_vew_grp['final_plant_date']], unit='days')
        df_vew_grp['START_DATE'] = df_vew_grp['final_plant_date']+timedelta
        
        # =============================================================================
        # Loop for each start_date   
        # =============================================================================
        list_df = []
        for start_date, df_vew_startdate_grp in tqdm(df_vew_grp.groupby(['START_DATE'])):
            
            # If start_date is out of raster_date's range, skip
            if not (raster_date[0]<=start_date<=raster_date[-1]):
                continue
            
            # Define image's indices (image[index, :, :])
            start_date_index = np.where(df_date['ideal']>=start_date)[0][0]
            
            # Define image's indices with windows (size = 2)
            '''
                Ex [2, 3, 0, 4, 5] << get raster band 2 3 4 5. 0 is raster band with missing date << to be replaced with nan
            '''
            available_bands = df_date.loc[start_date_index-2:start_date_index+2, 'raster_band']
            
            # Define which raster_date is not available (size = 2)
            '''
                Ex [False, False, True, False, False] 
            '''
            not_available_bands = ~df_date.loc[start_date_index-2:start_date_index+2, 'available']
            
            # In case of start_date too early
            if start_date_index < 2:
                available_bands = pd.concat([pd.Series([0]*(2-start_date_index)), available_bands])
                not_available_bands = pd.concat([pd.Series([True]*(2-start_date_index)), not_available_bands])
            
            # In case of start_date too late
            elif start_date_index > len(df_date)-3:
                available_bands = pd.concat([available_bands, pd.Series([0]*(start_date_index-len(df_date)+3))])
                not_available_bands = pd.concat([not_available_bands, pd.Series([True]*(start_date_index-len(df_date)+3))])                
            
            # Change image's indices to list
            available_bands = available_bands.to_list()
            not_available_bands = not_available_bands.to_list()

            # =============================================================================
            # Loop for each new_polygon_id and get flood pixel results
            # =============================================================================
            for i in range(len(df_vew_startdate_grp)):
                # Get new_polygon_id
                new_polygon_id = df_vew_startdate_grp.iat[i, 0]
                
                # Get rows, cols of 
                rows = []
                cols = []
                for item in df_mapping.at[new_polygon_id, 'row_col']:
                    rows.append(int(item.split('_')[0]))
                    cols.append(int(item.split('_')[1]))
                
                # Get flood pixels [indices, rows, cols]
                flood_details = raster_im[:, rows, cols][available_bands]
                
                # Change not available data row to nan (pick raster band 0 instead then replace with nan)
                flood_details[not_available_bands] = np.nan
                flood_details = flood_details.T
                
                # Create result's dataframe
                df = pd.DataFrame({
                                   'strip_id' : [strip_id]*len(flood_details),
                                   'new_polygon_id' : new_polygon_id,
                                   'PLANT_PROVINCE_CODE' : df_vew_startdate_grp.iat[i, 2],
                                   'PLANT_AMPHUR_CODE' : df_vew_startdate_grp.iat[i, 3],
                                   'PLANT_TAMBON_CODE' : df_vew_startdate_grp.iat[i, 4],
                                   'final_plant_date' : df_vew_startdate_grp.iat[i, 14],
                                   'START_DATE' : start_date,
                                   'loss_ratio' : df_vew_startdate_grp.iat[i, -1]
                                  })
                
                # Add flood pixel data
                for j in range(flood_details.shape[1]):
                    if j-2 == 0 :
                        df['t'] = flood_details[:, j]
                    else:
                        df[f"t{j-2:+d}"] = flood_details[:, j]
                
                # append dataframe to lis
                if len(df)!=0:
                    list_df.append(df)
                
        if len(list_df)!=0:
            # Combine all of the dataframes in list then save
            df = pd.concat(list_df, ignore_index=True)
            df.to_parquet(path_save)
#%%