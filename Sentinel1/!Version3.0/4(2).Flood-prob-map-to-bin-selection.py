import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_h5
from sklearn.metrics import confusion_matrix
from icedumpy.geo_tools import create_tiff, create_vrt
from icedumpy.eval_tools import kappa_from_cnf_matrix
#%%
def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%% Define constant parameters
root_raster_flood = r"G:\!PIER\!Flood-map\Sentinel-1"
root_roc_params = r"F:\CROP-PIER\CROP-WORK\Model\sentinel1"
#%% Define parameters
FPR = 0.2
sat_type = "S1AB"
#%%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    model_strip_id = strip_id # strip_id, "all"
    #%% Get paths from defined parameters and load (roc_params)
    path_roc_params = os.path.join(root_roc_params, sat_type, f"{strip_id}-modified", f"{strip_id}_RF_raw_pixel_values_roc_params.h5")
    root_raster_flood_prob = os.path.join(root_raster_flood, sat_type, "Prob", f"{strip_id}-modified")
    root_raster_flood_bins = os.path.join(root_raster_flood, sat_type, "Bins", f"{strip_id}-modified")
    dict_roc_params = load_h5(path_roc_params)
    threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold"], selected_fpr=FPR)
    # threshold_noloss = 0.2
    # threshold_loss = 0.7

    os.makedirs(root_raster_flood_bins, exist_ok=True)
    #%% Loop for each raster in root_raster_flood_prob
    for file in [file for file in os.listdir(root_raster_flood_prob) if file.endswith(".tif")]:
        path_raster_flood_prob = os.path.join(root_raster_flood_prob, file)
        path_raster_flood_bin = os.path.join(root_raster_flood_bins, f"Flood_map_{file}")
        if os.path.exists(path_raster_flood_bin):
            continue

        raster_flood_prob = rasterio.open(path_raster_flood_prob)
        raster_flood_prob_img = raster_flood_prob.read(1)
        row_nonrice, col_nonrice = np.where(raster_flood_prob_img == 0)
#%%
        raster_flood_bins_img = np.empty_like(raster_flood_prob_img, dtype="uint8")
        raster_flood_bins_img[raster_flood_prob_img < threshold_noloss] = 0
        raster_flood_bins_img[(threshold_noloss <= raster_flood_prob_img) & (raster_flood_prob_img <= threshold_loss)] = 1
        raster_flood_bins_img[raster_flood_prob_img > threshold_loss] = 2
        raster_flood_bins_img+=1
        raster_flood_bins_img[row_nonrice, col_nonrice] = 0

        create_tiff(path_save=path_raster_flood_bin,
                    im=raster_flood_bins_img,
                    projection=raster_flood_prob.crs.to_wkt(),
                    geotransform=raster_flood_prob.get_transform(),
                    drivername="GTiff",
                    nodata=0,
                    dtype="uint8"
                    )
        print(path_raster_flood_prob, path_raster_flood_bin)
    create_vrt(path_save=os.path.join(root_raster_flood_bins, "flood_bins.vrt"),
                list_path_raster=[os.path.join(root_raster_flood_bins, file) for file in os.listdir(root_raster_flood_bins) if file.endswith("tif")],
                list_band_name=[file.split(".")[0] for file in os.listdir(root_raster_flood_bins) if file.endswith("tif")],
                src_nodata=0,
                dst_nodata=0)
#%%
strip_id = "303"
model_strip_id = strip_id # strip_id, "all"
path_roc_params = os.path.join(root_roc_params, sat_type, f"{strip_id}-modified", f"{strip_id}_RF_raw_pixel_values_roc_params.h5")
root_raster_flood_prob = os.path.join(root_raster_flood, sat_type, "Prob", f"{strip_id}-modified")
root_raster_flood_bins = os.path.join(root_raster_flood, sat_type, "Bins", f"{strip_id}-modified")
dict_roc_params = load_h5(path_roc_params)
threshold = get_threshold_of_selected_fpr(dict_roc_params["fpr"], dict_roc_params["threshold"], selected_fpr=FPR)

y_train = dict_roc_params["y_train"]
y_predict_prob = dict_roc_params["y_predict_prob"]

list_stats = []
for threshold in np.arange(0.05, 1, 0.05):
    cnf_matrix = confusion_matrix(y_train, (y_predict_prob >= threshold).astype("uint8"))
    
    tpr = cnf_matrix[1, 1]/(cnf_matrix[1, 1]+cnf_matrix[1, 0])
    fpr = cnf_matrix[0, 1]/(cnf_matrix[0, 1]+cnf_matrix[0, 0])
    
    ppv = cnf_matrix[1, 1]/(cnf_matrix[1, 1]+cnf_matrix[0, 1])
    npv = cnf_matrix[0, 0]/(cnf_matrix[0, 0]+cnf_matrix[1, 0])
    
    f1_score = 2*((ppv*tpr)/(ppv+tpr))
    kappa = kappa_from_cnf_matrix(cnf_matrix)
    list_stats.append((threshold, tpr, ppv, fpr, npv, f1_score, kappa))
    print(f"{threshold:.2f}, {tpr:.2f}, {fpr:.2f}, {ppv:.2f}, {npv:.2f}, {f1_score:.4f}, {kappa:.4f}")
#%%
import pandas as pd
df_stats = pd.DataFrame(list_stats, columns=["threshold", "tps", "ppv", "fpr", "npv", "f1_score", "kappa"])
    
    
    

