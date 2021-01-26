import os
import sys
import json
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
import datetime
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from flood_fn import load_dataframe_mapping_vew, find_date_index, load_json, get_fuzzy_confusion_matrix
#%%
root_raster_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_flood_pixel"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
root_result = r"F:\CROP-PIER\CROP-WORK\Presentation\20200427"
#%%
# Initial variables for Image
raster_strip_id = 304
 
# Initial variables for evaluation
flood_map_window_size = 2

root_result = os.path.join(root_result, str(raster_strip_id))
if not os.path.exists(root_result):
    os.makedirs(os.path.join(root_result, 'result_polygon'))
#%%
path_threshold = r"F:\CROP-PIER\CROP-WORK\Flood-evaluation\\{0:d}\Flood_VV_S{1:d}_(0.80,0.20,0.70).json".format(raster_strip_id, raster_strip_id)
threshold_data = load_json(path_threshold)
df_mapping, df_vew = load_dataframe_mapping_vew(root_mapping, root_vew, root_raster_sentinel1, raster_strip_id, flood_map_window_size)
#%%
path_raster = os.path.join(root_raster_sentinel1, f"LSVVS{raster_strip_id}_2017-2019")
path_raster_flood = os.path.join(root_result, f"Flood_VV_S{raster_strip_id}.tiff")

raster = rasterio.open(path_raster)
raster_flood = rasterio.open(path_raster_flood)
raster_flood_im = raster_flood.read()
list_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
#%%  
ref = []
predicted = []   
for i in tqdm(range(len(df_vew))):
    df_vew_selected = df_vew.iloc[i]
    
    start_date = df_vew_selected['START_DATE']
    if start_date.year < 2017:
        break
    
    satellite_first_index = find_date_index(list_date, start_date-datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1
    satellite_last_index  = find_date_index(list_date, start_date+datetime.timedelta(12*flood_map_window_size)) # index start from 0, band start from 1

    satellite_first_index = max(0, satellite_first_index)
    satellite_last_index = min(satellite_last_index+1, len(raster_flood_im))  
    
    rows = []
    cols = []
    for item in df_mapping.at[df_vew_selected['new_polygon_id'], 'row_col']:
        rows.append(int(item.split('_')[0]))
        cols.append(int(item.split('_')[1]))     
    
    selected_pixels = raster_flood_im[satellite_first_index:satellite_last_index, rows, cols]  
    predicted_loss = np.any(selected_pixels, axis=0).astype('uint8')

    predicted_loss_ratio = predicted_loss.sum()/len(predicted_loss)
    ref_loss_ratio = df_vew_selected['loss_ratio']
 
    predicted.append(predicted_loss_ratio)
    ref.append(ref_loss_ratio)

f_cnf_matrix = get_fuzzy_confusion_matrix(predicted, ref)

# Calculate OA, DT, FA
OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall
FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1])
#%%
water_threshold = threshold_data['water_threshold']
to_water_threshold = threshold_data['change_to_water_threshold']
df_vew['predicted'] = predicted
df_vew['ref'] = ref
df_vew['result'] = np.nan
#%%
df_vew.loc[(df_vew['predicted'].astype('bool')==0) & (df_vew['ref'].astype('bool')==0), 'result'] = 'TN'
df_vew.loc[(df_vew['predicted'].astype('bool')==0) & (df_vew['ref'].astype('bool')==1), 'result'] = 'FN'
df_vew.loc[(df_vew['predicted'].astype('bool')==1) & (df_vew['ref'].astype('bool')==1), 'result'] = 'TP'
df_vew.loc[(df_vew['predicted'].astype('bool')==1) & (df_vew['ref'].astype('bool')==0), 'result'] = 'FP'
#%%
raster_im = raster.read()
root_graph = os.path.join(root_result, 'graph')
if not os.path.exists(root_graph):
    os.mkdir(root_graph)
#%%
for result, df_vew_grp in df_vew.groupby(['result']):
    print(f"Start: {result}")
    root_graph_each = os.path.join(root_graph, result)
    if not os.path.exists(root_graph_each):
        os.mkdir(root_graph_each)
#%%    
    for (new_polygon_id, start_date), df_vew_grp_selected in df_vew_grp.groupby(['new_polygon_id', 'START_DATE']):
        plt.close('all')
        print(new_polygon_id, start_date)
        df_vew_grp_selected = df_vew_grp_selected.iloc[0]
        
        satellite_index = find_date_index(list_date, start_date) # index start from 0, band start from 
        
 
        
        selected_pixels = raster_im[:, rows, cols]  
        selected_pixels_mean = np.mean(selected_pixels, axis=1)
    
        lower_limit = max(0, satellite_index-8)
        upper_limit = min(satellite_index+8, len(list_date))
    
        fig, ax = plt.subplots(2)

        ax[0].plot(list_date[lower_limit:upper_limit], selected_pixels_mean[lower_limit:upper_limit])
        ax[1].plot(list_date[lower_limit:upper_limit], -np.diff(np.hstack((selected_pixels_mean[lower_limit], selected_pixels_mean[lower_limit:upper_limit]))))
        
        # Set x label 
        ax[0].set_xticks([item.date() for item in list_date[lower_limit:upper_limit]])
        ax[0].set_xticklabels([item.date() for item in list_date[lower_limit:upper_limit]])
        ax[1].set_xticks([item.date() for item in list_date[lower_limit:upper_limit]])
        ax[1].set_xticklabels([item.date() for item in list_date[lower_limit:upper_limit]])
        
        # Draw start date vertical line
        ax[0].axvline(x=start_date, color='r', linestyle='dashed', linewidth=1)
        ax[1].axvline(x=start_date, color='r', linestyle='dashed', linewidth=1)
        
        # Draw plant date vertical line
        ax[0].axvline(x=df_vew_grp_selected['final_plant_date'], color='g', linestyle='dashed', linewidth=1)
        ax[1].axvline(x=df_vew_grp_selected['final_plant_date'], color='g', linestyle='dashed', linewidth=1)
        
        # Draw sampling date vertical line
        ax[0].axvline(x=list_date[satellite_index], color='b', linestyle='dashed', linewidth=1)
        ax[1].axvline(x=list_date[satellite_index], color='b', linestyle='dashed', linewidth=1)
        
        # Draw water threshold, change threshold horizontal line
        ax[0].axhline(y=water_threshold, color='b', linestyle='-', linewidth=1)
        ax[1].axhline(y=to_water_threshold, color='b', linestyle='-', linewidth=1)
        
        # Write water threshold, change threshold values
        ax[0].text(start_date, water_threshold, 'water threshold = {0:.4f}'.format(water_threshold))
        ax[1].text(start_date, to_water_threshold, 'change to water threshold = {0:.4f}'.format(to_water_threshold))
        
        # Write "start date" at start date vertical line
        ax[0].text(start_date, ax[0].get_ylim()[1], 'start date', color='r', horizontalalignment='center')
        ax[1].text(start_date, ax[1].get_ylim()[1], 'start date', color='r', horizontalalignment='center')
        
        # Write "plant date" at plant date vertical line
        ax[0].text(df_vew_grp_selected['final_plant_date'], ax[0].get_ylim()[1], 'plant date', color='g', horizontalalignment='center')
        ax[1].text(df_vew_grp_selected['final_plant_date'], ax[1].get_ylim()[1], 'plant date', color='g', horizontalalignment='center')
        
        # Write "t" at sampling date vertical line
        ax[0].text(list_date[satellite_index], ax[0].get_ylim()[1], 't', color='b', horizontalalignment='center')
        ax[1].text(list_date[satellite_index], ax[1].get_ylim()[1], 't', color='b', horizontalalignment='center')
        
        # Write value to each point
        for date, mean in zip(list_date[lower_limit:upper_limit], selected_pixels_mean[lower_limit:upper_limit]):
            ax[0].text(date, mean, '{0:4f}'.format(mean), horizontalalignment='center', verticalalignment='center')
        for date, mean in zip(list_date[lower_limit:upper_limit], -np.diff(np.hstack((selected_pixels_mean[lower_limit], selected_pixels_mean[lower_limit:upper_limit])))):
            ax[1].text(date, mean, '{0:4f}'.format(mean), horizontalalignment='center', verticalalignment='center')
            
        # Write xy label 
        ax[0].set_xlabel('date')
        ax[0].set_ylabel('pixel value')
        
        # Write xy label
        ax[1].set_xlabel('date')
        ax[1].set_ylabel('change value')
        
        # Increase figure size
        fig.set_size_inches(14, 10)
        fig.suptitle(f'new Polygon id: {new_polygon_id}, loss_ratio(actual): {df_vew_grp_selected["loss_ratio"]:.4f}, loss_ratio(predicted) : {df_vew_grp_selected["predicted"]:.4f}', fontsize=16)
        
        # Save fig
        fig.savefig(os.path.join(root_graph_each, f"{new_polygon_id}_{start_date.date()}.png"), dpi=150, transparent=True)
#%%
