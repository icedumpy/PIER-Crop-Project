import os
import flood_fn
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
index_name = "s204"
root_flood_map = os.path.join(r"F:\CROP-PIER\CROP-WORK\Flood-map\one_data_point", index_name[1:])
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
#%%
flood_raster = gdal.Open(os.path.join(root_flood_map, f"Flood_LSVV{index_name.upper()}_2017-2019_log_likelihood"))
flood_raster_date = flood_fn.get_raster_date(flood_raster, dtype='gdal')
flood_im = flood_raster.ReadAsArray()
#%%
df_vew, df_mapping = flood_fn.load_vew_mapping_df(root_mapping, root_vew, index_name, date_range=(flood_raster_date[0], flood_raster_date[-1]))

# Randomly add 1-4 months date into final_plant_date (only no flood) then fill in start_date (not more than the last raster date)
df_vew = df_vew[(((flood_raster_date[-1] - df_vew['final_plant_date']).dt.days) >= 30) | (~pd.isnull(df_vew['START_DATE']))]
timedelta = pd.to_timedelta([np.random.randint(30, min(121, (flood_raster_date[-1]-date).days+1)) for date in df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date']], unit='days')
df_vew.at[df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date'].index, 'START_DATE'] = df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date']+timedelta  
#%%
# Get log_prob from every pixel
list_predicted_pixels = []
list_ref_loss_ratio = []
for start_date, df_vew_grp in df_vew.groupby(['START_DATE']):
    flood_index = np.where(start_date<=flood_raster_date)[0][0]
    
    # Check if flood image is not empty
    if not ((flood_im[flood_index] == -9999).all()):
        for index in df_vew_grp.index:
            new_polygon_id = df_vew_grp.loc[index, 'new_polygon_id']
            ref_loss_ratio = df_vew_grp.loc[index, 'loss_ratio']
            
            # Get row, col of polygon
            rows = []
            cols = []
            for item in df_mapping.at[new_polygon_id, 'row_col']:
                rows.append(int(item.split('_')[0]))
                cols.append(int(item.split('_')[1]))
            
            predicted_pixels = flood_im[flood_index, rows, cols]
            
            list_predicted_pixels.append(predicted_pixels)
            list_ref_loss_ratio.append(ref_loss_ratio)
            
all_predicted_pixels = np.concatenate(list_predicted_pixels)

tqdm.write(f"Thresholding for ROC Curve")
# Find DT(tpr), FA(fpr) for every treshold
thresholds = np.insert(np.linspace(all_predicted_pixels.min(), all_predicted_pixels.max(), 200)[::-1], 0, all_predicted_pixels.max()+1)

# Loop for each threshold
tpr = []
fpr = []
for threshold in thresholds:
    list_predicted_loss_ratio = []

    # Loop for each predicted plot
    for predicted_pixels in list_predicted_pixels:
        predicted_pixels = (predicted_pixels>=threshold).astype('uint8')
        predicted_loss_ratio = predicted_pixels.sum()/len(predicted_pixels)
        
        list_predicted_loss_ratio.append(predicted_loss_ratio)
  
    f_cnf_matrix = flood_fn.get_fuzzy_confusion_matrix(list_predicted_loss_ratio, list_ref_loss_ratio)
    OA = 100*np.diag(f_cnf_matrix).sum()/len(list_predicted_loss_ratio)
    DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall, TPR
    FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1]) # FPR
    
    tpr.append(DT)
    fpr.append(FA)        
#%%
plt.plot(fpr, tpr, "b-", fpr, fpr, "--")
plt.xlabel("False Alarm")
plt.ylabel("Detection")
plt.grid()
plt.show()
#%%
#np.save(os.path.join(root_flood_map, 'tpr.npy'), tpr)
#np.save(os.path.join(root_flood_map, 'fpr.npy'), fpr)
#np.save(os.path.join(root_flood_map, 'thresholds.npy'), thresholds)
#%%







