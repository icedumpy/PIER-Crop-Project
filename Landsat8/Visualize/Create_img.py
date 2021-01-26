import os
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import icedumpy
tqdm.pandas()

label_color = {"TP" : "blue",
               "TN" : "green",
               "FP" : "red",
               "FN" : "yellow"}

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

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def load_color_infrared_images(root_raster, dict_img, channel):
    # Delete dict_img[key] that key not in range of (channel-2, channel+2)
    [dict_img.pop(key) for key in [key for key in dict_img.keys() if not key in range(channel-2, channel+2)]]
    
    # Load 4 infrared images into dict_img (channel-2, channel-1, channel, channel+1)
    for channel_in_range in range(channel-2, channel+2):
        if channel_in_range in dict_img.keys():
            continue
        tqdm.write(f"Loading channel: {channel_in_range}")
        img = icedumpy.geo_tools.create_landsat8_color_infrared_img(root_raster, channel_in_range, "rgb", channel_first=False)
        dict_img[channel_in_range] = img
    return dict_img

def initialize_grid_figure(figsize=(10, 10)):
    fig = plt.figure(figsize=figsize) # Notice the equal aspect ratio
    ax = [fig.add_subplot(2, 2, i+1) for i in range(4)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(axis='both', which='both', length=0)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    return fig, ax
#%% Initialize parameters
pathrow = "129049"

root_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_polygon_id_rowcol_map_prov_scene_merged_v2"  

root_model_visualize = r"F:\CROP-PIER\CROP-WORK\Model_visualization\model_129048_129049_130048_130049_B2_B3_B4_B5"
root_raster = rf"F:\CROP-PIER\CROP-WORK\LS8_VRT\{pathrow}"
root_raster_flood =  rf"F:\CROP-PIER\CROP-WORK\Model_visualization\Flood_map\Landsat-8\{pathrow}"
#%% Get raster's details of the selected pathrow (Want crs)
raster = rasterio.open(os.path.join(root_raster, "ls8_129049_B5_TOA.vrt"))
raster_date = icedumpy.geo_tools.get_raster_date(raster, "datetime", "rasterio")
raster_crs = raster.crs.to_dict()
raster_transform = raster.transform

# Get flood maps
# Load raster of flood map of selected pathrow into list
dict_flood_map = dict()
for file in os.listdir(root_raster_flood):
    if not "." in file:
        dict_flood_map[int(file.split("_")[1][1:])] = rasterio.open(os.path.join(root_raster_flood, file))
#%% Load data (Only the test set)
df_model_result = load_df_model_result(root_model_visualize)
df_model_result = pd.read_parquet(os.path.join(root_model_visualize, "x.parquet"))       
df_model_result = df_model_result[df_model_result['pathrow']==str(pathrow)]
df_model_result = df_model_result.assign(raster_date_index=df_model_result['START_DATE'].progress_apply(lambda val: np.nonzero((val <= raster_date))[0][0]))

# Show stats
try:
    temp = df_model_result.groupby(['pathrow', 'predict_label']).size()["129049"]
    print("DT:", 100*temp['TP']/(temp['TP']+temp['FN']))
    print("FA:", 100*temp['FP']/(temp['FP']+temp['TN']))
    print("OA:", 100*(temp['TP']+temp['TN'])/(temp.sum()))
except:
    model = icedumpy.io_tools.load_model(os.path.join(root_model_visualize, "model.joblib")) 
    y_pred_proba = model.predict_proba(df_model_result.iloc[:, 10:-2].values)
    roc_params = icedumpy.io_tools.load_h5(os.path.join(root_model_visualize, "roc_params.h5"))
    thresh = get_threshold_of_selected_fpr(roc_params["fpr_test"], roc_params["thresholds_test"], selected_fpr=0.3)
    df_model_result['pred'] = (y_pred_proba[:, 1]>=thresh).astype('uint8')
    
    df_model_result['predict_label'] = None
    df_model_result.loc[(df_model_result['label'] & df_model_result['pred']).astype('bool'), "predict_label"] = "TP"
    df_model_result.loc[(~df_model_result['label'].astype('bool') & ~df_model_result['pred']).astype('bool'), "predict_label"] = "TN"
    df_model_result.loc[(~df_model_result['label'].astype('bool') & df_model_result['pred']).astype('bool'), "predict_label"] = "FP"
    df_model_result.loc[(df_model_result['label'].astype('bool') & ~df_model_result['pred']).astype('bool'), "predict_label"] = "FN"
    
    temp = df_model_result.groupby(['pathrow', 'predict_label']).size()["129049"]
    print("DT:", 100*temp['TP']/(temp['TP']+temp['FN']))
    print("FA:", 100*temp['FP']/(temp['FP']+temp['TN']))
    print("OA:", 100*(temp['TP']+temp['TN'])/(temp.sum()))
#%% Load shapefiles one pathrow contains many province
list_p = [file.split("_")[-2] for file in os.listdir(root_mapping) if file.split(".")[0].split("_")[-1] == pathrow]
list_df_vew = []

# Load and filter by the test set "new_polygon_id" set
for file in os.listdir(root_vew):
    if file.split(".")[0][-3:] in list_p:
        df_vew_temp = pd.read_parquet(os.path.join(root_vew, file))
        df_vew_temp = df_vew_temp[df_vew_temp['new_polygon_id'].isin(df_model_result['new_polygon_id'])]
        list_df_vew.append(df_vew_temp)
df_vew = pd.concat(list_df_vew)
df_vew['final_plant_date'] = pd.to_datetime(df_vew['final_plant_date'])

# Convert df to gdf
gdf_vew = icedumpy.geo_tools.convert_to_geodataframe(df_vew, to_crs=raster_crs)
gdf_vew = gdf_vew[(gdf_vew['DANGER_TYPE_NAME']=="อุทกภัย") | pd.isnull(gdf_vew['DANGER_TYPE_NAME'])]
gdf_vew = gdf_vew.drop(columns = ['START_DATE'])
gdf_vew = pd.merge(gdf_vew, df_model_result[['new_polygon_id', 'final_plant_date', 'START_DATE', 'raster_date_index']], on=['new_polygon_id', 'final_plant_date'], how='inner')
#%%
dict_img = dict()

# For flood raster
df_date_raster_channel = pd.DataFrame(raster_date, columns=['date'])
df_date_raster_channel['normal_raster_channel'] = df_date_raster_channel.index+1
df_date_raster_channel['flood_raster_channel'] = 0
for year, df_temp in df_date_raster_channel.groupby(df_date_raster_channel['date'].dt.year):
    df_date_raster_channel.loc[df_temp.index, 'flood_raster_channel'] = np.arange(1, len(df_temp)+1, 1)
#%%
figsize = (10, 10)
for raster_date_index, df_model_result_selected_raster_date_index in tqdm(df_model_result.groupby(['raster_date_index'])):
    channel = raster_date_index+1
    tqdm.write(f"{channel}, {raster_date[raster_date_index].date()}")
    
    # =============================================================================
    # Step 1. Load raster images and pre-compute color infrared images
    # =============================================================================
    dict_img = load_color_infrared_images(root_raster, dict_img, channel)
    
    # =============================================================================
    # Step 2. Plot raster infrared images    
    # =============================================================================
    # Initialize plot figure1 (grid for 4(2x2) images)
    fig1, ax = initialize_grid_figure(figsize=figsize) # Top-left, Top-right, Bottom-left, Bottom-right
    
    # Assign each infrared image to each grid 
    # [1 2] 
    # [3 4]
    rasterio.plot.show(dict_img[channel-2], ax=ax[0], transform=raster_transform)
    rasterio.plot.show(dict_img[channel-1], ax=ax[1], transform=raster_transform)
    rasterio.plot.show(dict_img[channel  ], ax=ax[2], transform=raster_transform)
    rasterio.plot.show(dict_img[channel+1], ax=ax[3], transform=raster_transform)
    fig1.show()
    fig1.savefig(rf"C:\Users\PongporC\Desktop\14-7-2020\{raster_date[raster_date_index].date()}_img.png", dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
    fig1.clf()
    plt.close()
    
    # =============================================================================
    # Step 3. Load, Plot raster flood image and plot shapefiles
    # =============================================================================
    # Load raster flood  
    raster_flood = dict_flood_map[raster_date[channel-1].year]
    flood_img = raster_flood.read(int(df_date_raster_channel.loc[df_date_raster_channel['normal_raster_channel'] == channel, 'flood_raster_channel'].values[0]))
    flood_img = np.where(flood_img==raster_flood.nodata, np.nan, flood_img)
    
    # Plot flood image on figure2
    fig2 = plt.figure(figsize=figsize)
    ax = fig2.add_axes([0.1, 0.1, 0.8, 0.8],
                       xticklabels=[], 
                       yticklabels=[],
                      )
    ax.axis('off')
    
    rasterio.plot.show(flood_img, ax=ax, transform=raster_transform, cmap='jet_r')
    
    # Plot shapefiles
    gdf_vew.loc[gdf_vew['raster_date_index']==raster_date_index].loc[pd.isna(gdf_vew["DANGER_TYPE_NAME"])].boundary.plot(ax=ax, edgecolor='green', alpha=0.2)
    gdf_vew.loc[gdf_vew['raster_date_index']==raster_date_index].loc[~pd.isna(gdf_vew["DANGER_TYPE_NAME"])].boundary.plot(ax=ax, edgecolor='violet', alpha=0.1)
    
    fig2.show()
    fig2.savefig(rf"C:\Users\PongporC\Desktop\14-7-2020\{raster_date[raster_date_index].date()}_flood.png", dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
    fig2.clf()
    plt.close()
#%%
for pathrow, temp in df_model_result.groupby(['pathrow']):
    print(pathrow, len(temp))
    print("DT:", 100*temp['predict_label'].value_counts()['TP']/(temp['predict_label'].value_counts()['TP']+temp['predict_label'].value_counts()['FN']))
    print("MR:", 100*temp['predict_label'].value_counts()['FN']/(temp['predict_label'].value_counts()['FN']+temp['predict_label'].value_counts()['TP']))
    print("FA:", 100*temp['predict_label'].value_counts()['FP']/(temp['predict_label'].value_counts()['FP']+temp['predict_label'].value_counts()['TN']))
    print("OA:", 100*(temp['predict_label'].value_counts()['TP']+temp['predict_label'].value_counts()['TN'])/(temp['predict_label'].value_counts().sum()))
    print()
#%%
ice = df_model_result[df_model_result['new_polygon_id']==324675]