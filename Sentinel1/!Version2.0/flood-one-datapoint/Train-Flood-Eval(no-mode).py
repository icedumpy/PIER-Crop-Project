import os
import numpy as np
import pandas as pd
import flood_fn
from osgeo import gdal
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
#index_names = ['s109', 's204', 's205', 's302', 's303', 's304', 's402', 's403']
index_names = [101, 102, 103, 104, 105, 106, 107, 108, 109, 
               201, 202, 203, 204, 205, 206, 207, 208,
               301, 302, 303, 304, 305, 306,
               401, 402, 403]
index_names = [f's{item}' for item in index_names]
root_df_dry = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_noflood_pixel_from_mapping_v3"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_flood_pixel_from_mapping_v3"
root_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_model = r"F:\CROP-PIER\CROP-WORK\Model"
root_flood_map = r"F:\CROP-PIER\CROP-WORK\Flood-map"
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
#%%
for index_name in tqdm(index_names):
    
    # =============================================================================
    # Train and save model
    # =============================================================================
    tqdm.write(f"START: {index_name}")
    tqdm.write(f"Training")
    
    # Define model to be used
    # =============================================================================
    model_name = 'all'
    # =============================================================================
    
    path_model = os.path.join(root_model, model_name+".joblib")
    
    # Train or load model
    if not os.path.exists(path_model):
        df_dry, df_flood = flood_fn.load_dataset_df(root_df_dry, root_df_flood, model_name)
        model, [x_train, x_test, y_train, y_test] = flood_fn.train_model(df_dry, df_flood, balanced=True, verbose=0)

        # Save model
        flood_fn.save_model(model, path_model)
        
    else:
        model = flood_fn.load_model(path_model)
        
    # =============================================================================
    # Create flood map
    # =============================================================================
    tqdm.write(f"Create flood map")
    path_raster = os.path.join(root_raster, f"LSVV{index_name.upper()}_2017-2019")
    path_mask = os.path.join(root_flood_map, index_name.replace('s', ''), 'rice_mask.shp')
    
    if index_name==model_name:
        path_flood_im = os.path.join(root_flood_map, index_name.replace('s', ''), f"Flood_LSVV{index_name.upper()}_2017-2019_log_likelihood")
    else:
        path_flood_im = os.path.join(root_flood_map, index_name.replace('s', ''), f"Flood_LSVV{index_name.upper()}_2017-2019_log_likelihood_all")

    if not os.path.exists(path_flood_im):
        flood_im, raster_date =  flood_fn.create_flood_map(path_flood_im, model, path_raster, path_mask, root_mapping, root_vew, index_name)    
    else:
        flood_raster = gdal.Open(path_flood_im)
        flood_im = flood_raster.ReadAsArray()
        raster_date = flood_fn.get_raster_date(flood_raster, dtype='gdal')
    
    # =============================================================================
    # Eval
    # =============================================================================
    tqdm.write(f"Start Evaluate")

    # Load vew
    tqdm.write(f"Loading data for evaluate")
    df_vew, df_mapping = flood_fn.load_vew_mapping_df(root_mapping, root_vew, index_name, date_range=(raster_date[0], raster_date[-1]))
    
    # Randomly add 1-4 months date into final_plant_date (only no flood) then fill in start_date (not more than the last raster date)
    df_vew = df_vew[(((raster_date[-1] - df_vew['final_plant_date']).dt.days) >= 30) | (~pd.isnull(df_vew['START_DATE']))]
    timedelta = pd.to_timedelta([np.random.randint(30, min(121, (raster_date[-1]-date).days+1)) for date in df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date']], unit='days')
    df_vew.at[df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date'].index, 'START_DATE'] = df_vew[pd.isnull(df_vew['START_DATE'])]['final_plant_date']+timedelta      
    
    tqdm.write(f"Get log proba values of each plot")
    
    # Get log_prob from every pixel
    list_predicted_pixels = []
    list_ref_loss_ratio = []
    for start_date, df_vew_grp in df_vew.groupby(['START_DATE']):
        flood_index = np.where(start_date<=raster_date)[0][0]
        
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

    # =============================================================================
    # Show result and save
    # =============================================================================
    tqdm.write(f"Showing result")
    # Plot ROC
    plt.close('all')
    plt.figure(figsize=(16.0, 9.0))
    plt.plot(fpr, tpr, "b-", fpr, fpr, "--")
    plt.xlabel("False Alarm")
    plt.ylabel("Detection")             
    plt.grid()
    plt.title(f"Scene: {index_name[1:]}")
    plt.xlim(left=-5, right=105)
    plt.ylim(bottom=-5, top=105)
    
    # Define file name
    tqdm.write(f"Save results")
    if model_name == index_name:
        lastname = f"_{model_name}"
    else:
        lastname = f"_{index_name}_{model_name}"
        
    tqdm.write(f"Saving result")
    # Save result
    plt.savefig(os.path.join(root_flood_map, index_name[1:], f'ROC{lastname}.png'), transparent=True, dpi=200)    
    np.save(os.path.join(root_flood_map, index_name[1:], f'tpr{lastname}.npy'), tpr)
    np.save(os.path.join(root_flood_map, index_name[1:], f'fpr{lastname}.npy'), fpr)
    np.save(os.path.join(root_flood_map, index_name[1:], f'thresholds{lastname}.npy'), thresholds)
#%%