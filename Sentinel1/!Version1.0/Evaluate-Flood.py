import os
import json
import numpy as np
from itertools import product 
from tqdm import tqdm
import flood_fn
#%%
root_raster_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_flood_pixel"
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v3"

root_result = r"F:\CROP-PIER\CROP-WORK\Flood-evaluation"
#root_result = r"C:\Users\PongporC\Desktop\นาปี"
#root_result = r"C:\Users\PongporC\Desktop\นาปรังปินาด"

if not os.path.exists(root_result):
    os.mkdir(root_result)
#%%
# Inital variables of flood parameters >> For Image    
index_name = 's401'
window_size = 2

array_percentiles = np.arange(0.5, 1.00, 0.1)  # For n loop (Start, Stop, Step), For one loop (Start, Start+Step/2, Step)

# Initial variables for Image
raster_strip_id = 402
selected_band = (0, 0)

# Initial variables for evaluation
flood_map_window_size = 2
#%% Create initial variable
filename_base = f"Flood_VV_S{raster_strip_id}"
#%% Load initial data (only once)
# Parameters
df_flood = flood_fn.load_dataframe_flood(root_df_flood, index_name)

# Evaluation 
df_mapping, df_vew = flood_fn.load_dataframe_mapping_vew(root_mapping, root_vew, root_raster_sentinel1, raster_strip_id, flood_map_window_size=flood_map_window_size)

# นาปรัง
#df_vew = df_vew[~((df_vew['final_plant_date'].dt.month>=5) & (df_vew['final_plant_date'].dt.month<=10))]

#%%
#threshold_parameters = np.array((list(product(array_percentiles, repeat=3))))
threshold_parameters = np.array((list(product(array_percentiles, repeat=3))))[1::2]
#%%
for water_percentile, change_to_water_percentile, change_from_water_percentile in tqdm(threshold_parameters):
    change_to_water_percentile = 1-change_to_water_percentile

    tqdm.write(f"Start({water_percentile:.2f}, {change_to_water_percentile:.2f}, {change_from_water_percentile:.2f})")

    # =============================================================================
    # Get flood parameters, and save 
    # =============================================================================    
    water_threshold, change_to_water_threshold, change_from_water_threshold = flood_fn.get_flood_parameters(df_flood, water_percentile, change_to_water_percentile, change_from_water_percentile)

    # Create dictionary of threshold data
    threshold_data = {'raster_strip_id' : raster_strip_id,
                      'parameters_from' : index_name,
                      'water_percentile' : water_percentile,
                      'change_to_water_percentile' : change_to_water_percentile,
                      'change_from_water_percentile' : change_from_water_percentile,
                      'water_threshold' : water_threshold,
                      'change_to_water_threshold' : change_to_water_threshold,
                      'change_from_water_threshold' : change_from_water_threshold
                     }
    # =============================================================================
    # Save threshold parameters
    # =============================================================================
    path_threshold_data = flood_fn.save_parameters(threshold_data, filename_base, root_result, raster_strip_id)     
    if os.path.exists(os.path.join(root_result, f'{raster_strip_id}', os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy')):
        continue                       
    tqdm.write(f"{path_threshold_data} Saved")

    # =============================================================================
    #     Create flood map
    # =============================================================================
    path_raster = os.path.join(root_raster_sentinel1, f"LSVVS{raster_strip_id}_2017-2019")
    
    if not 'init_flood' in locals():
        map_flood, list_date, init_flood = flood_fn.creat_flood_image(path_raster, threshold_data, selected_band = selected_band)

    else:
        map_flood, list_date, _ = flood_fn.creat_flood_image(path_raster, threshold_data, selected_band = selected_band, init=init_flood)
    
    # =============================================================================
    #     Evaluation
    # =============================================================================
        
    # Get predicted and ref loss ratio
    predicted, ref = flood_fn.get_predict_and_ref(df_mapping, df_vew, map_flood, list_date, init_flood, flood_map_window_size)
    
    # Create Fuzzy Confusion Matrix
    f_cnf_matrix = flood_fn.get_fuzzy_confusion_matrix(predicted, ref)
    
    # Calculate OA, DT, FA
    OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
    DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # aka recall
    FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1])
    
    assert OA>=0
    assert DT>=0
    assert FA>=0
    
    results = { 
                'OA' : OA,
                'DT' : DT,
                'FA' : FA,  
                'total_flood' : float((np.array(ref)>0).sum()),
                'total_no_flood' : float((np.array(ref)==0).sum())
              }
    
    json.dump(results, open(os.path.join(root_result, f"{raster_strip_id}", os.path.basename(path_threshold_data).split('.json')[0]+'_result.json'), 'w'))
    tqdm.write(f"{os.path.join(root_result, str(raster_strip_id), os.path.basename(path_threshold_data).split('.json')[0]+'_result.json')} Saved")
    np.save(os.path.join(root_result, f"{raster_strip_id}", os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy'), f_cnf_matrix) 
    tqdm.write(f"{os.path.join(root_result, str(raster_strip_id), os.path.basename(path_threshold_data).split('.json')[0]+'_result_fcnf.npy')} Saved")
    tqdm.write(f"Detection rate: {results['DT']}")
    tqdm.write(f"False alarm rate: {results['FA']}")
    tqdm.write("")
#%%
##%%
for water_percentile, change_to_water_percentile, change_from_water_percentile in tqdm(threshold_parameters):
    change_to_water_percentile = 1-change_to_water_percentile

    tqdm.write(f"Start({water_percentile:.2f}, {change_to_water_percentile:.2f}, {change_from_water_percentile:.2f})")

    # =============================================================================
    # Get flood parameters, and save 
    # =============================================================================    
    water_threshold, change_to_water_threshold, change_from_water_threshold = flood_fn.get_flood_parameters(df_flood, water_percentile, change_to_water_percentile, change_from_water_percentile)
    print(water_threshold, change_to_water_threshold, change_from_water_threshold)