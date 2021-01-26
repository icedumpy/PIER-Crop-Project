import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave
import icedumpy

tqdm.pandas()

def load_df_model_result(root):
    df_TP = pd.read_parquet(os.path.join(root, "df_TP.parquet"))
    df_TP = df_TP.assign(predict_label = "TP")
    
    df_TN = pd.read_parquet(os.path.join(root, "df_TN.parquet"))
    df_TN = df_TN.assign(predict_label = "TN")

    df_FN = pd.read_parquet(os.path.join(root, "df_FN.parquet"))
    df_FN = df_FN.assign(predict_label = "FN")

    df_FP = pd.read_parquet(os.path.join(root, "df_FP.parquet"))  
    df_FP = df_FP.assign(predict_label = "FP")
    
    df_model_result = pd.concat((df_TP, df_TN, df_FN, df_FP), ignore_index=True)
    return df_model_result

def create_color_infrared_img(root_ls8, pathrow, channel):
    raster_r = rasterio.open(os.path.join(root_ls8, str(pathrow), f"ls8_{pathrow}_B5_TOA.vrt"))
    raster_g = rasterio.open(os.path.join(root_ls8, str(pathrow), f"ls8_{pathrow}_B4_TOA.vrt"))
    raster_b = rasterio.open(os.path.join(root_ls8, str(pathrow), f"ls8_{pathrow}_B3_TOA.vrt"))
    color_infrared_img = np.vstack((raster_r.read([channel]), raster_g.read([channel]), raster_b.read([channel])))
    color_infrared_img = np.moveaxis(color_infrared_img, 0, -1)
    return color_infrared_img
#%%
root_ls8 = r"F:\CROP-PIER\CROP-WORK\LS8_VRT"

root_model_visualize = r"F:\CROP-PIER\CROP-WORK\Model visualization\model_127048_127049_128048_128049_B2_B3_B4_B5"

label_values = {"TP" : 1,
                "TN" : 2,
                "FP" : 3,
                "FN" : 4}
#%% Execute
# Load model result dataframe (Correct, miss, false alarm prediction)
df_model_result = load_df_model_result(root_model_visualize)
print(df_model_result.groupby(['pathrow', 'predict_label']).size())
#%%
# Filter by selected pathrow
pathrow = 127049
root_save = os.path.join(root_model_visualize, str(pathrow))
df_model_result_selected_pathrow = df_model_result[df_model_result['pathrow']==str(pathrow)]

# Get raster_date of selected pathrow
raster_temp = rasterio.open(os.path.join(root_ls8, str(pathrow), f"ls8_{pathrow}_B1_TOA.vrt"))
raster_date = icedumpy.geo_tools.get_raster_date(raster_temp, datetimetype='datetime', dtype='rasterio')

# Add raster_date_index (of start date) column
df_model_result_selected_pathrow = df_model_result_selected_pathrow.assign(raster_date_index=df_model_result_selected_pathrow['START_DATE'].progress_apply(lambda val: np.nonzero((val <= raster_date))[0][0]))
#%%
for raster_date_index, df_model_result_selected_pathrow_raster_date_index in tqdm(df_model_result_selected_pathrow.groupby(['raster_date_index'])):
    if len(df_model_result_selected_pathrow_raster_date_index)<8000:
        continue
    dir_save = os.path.join(root_save, str(raster_date[raster_date_index].date()))
    os.makedirs(dir_save, exist_ok=True)
    
    # Save label image
    path_save = os.path.join(dir_save, "label.png")
    predict_label_img = np.zeros((raster_temp.height, raster_temp.width), dtype='uint8')
    for key in label_values.keys():
        rows = df_model_result_selected_pathrow_raster_date_index[df_model_result_selected_pathrow_raster_date_index['predict_label']==key]['row']
        cols = df_model_result_selected_pathrow_raster_date_index[df_model_result_selected_pathrow_raster_date_index['predict_label']==key]['col']
        predict_label_img[rows, cols] = label_values[key]
    imsave(path_save, predict_label_img, check_contrast=False)
    
    # Save raster image
    start_index = raster_date_index-2
    stop_index = raster_date_index+2
    for i in range(start_index, stop_index):
        tqdm.write(f"{raster_date[raster_date_index].date()} << {raster_date[i].date()}")
        path_save = os.path.join(dir_save, str(raster_date[i].date())+".png")
        if os.path.exists(path_save):
            continue
        
        channel = i+1 # Channel(band) of raster start from 1 to N but i start from 0 to N-1, N the total number of channels
        color_infrared_img = create_color_infrared_img(root_ls8, channel)
        color_infrared_img = (255*color_infrared_img).astype('uint8')
        
        imsave(path_save, color_infrared_img, check_contrast=False)
    tqdm.write("")
#%%
