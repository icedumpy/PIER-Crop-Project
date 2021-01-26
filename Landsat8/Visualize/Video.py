import os
import cv2
import numpy as np
import pandas as pd
import rasterio
import icedumpy
import matplotlib.pyplot as plt
#%% Define parameters
pathrow = 129049
root_raster = rf"F:\CROP-PIER\CROP-WORK\LS8_VRT\{pathrow}"
root_flood_map = rf"F:\CROP-PIER\CROP-WORK\Flood_map\Landsat-8\{pathrow}" 
#%% Load initialize parameters
# Create empty dictionaty (for storing raster images)
dict_img = dict()

# Load raster of flood map of selected pathrow into list
dict_flood_map = dict()
for file in os.listdir(root_flood_map):
    if not "." in file:
        dict_flood_map[int(file.split("_")[1][1:])] = rasterio.open(os.path.join(root_flood_map, file))

# Get raster date of landsat8 of selected pathrow
raster_temp = rasterio.open(os.path.join(root_raster, os.listdir(root_raster)[0]))
date_raster = icedumpy.geo_tools.get_raster_date(raster_temp, 'datetime', 'rasterio').tolist()

# For flood raster
df_date_raster_channel = pd.DataFrame(date_raster, columns=['date'])
df_date_raster_channel['normal_raster_channel'] = df_date_raster_channel.index+1
df_date_raster_channel['flood_raster_channel'] = 0
for year, df_temp in df_date_raster_channel.groupby(df_date_raster_channel['date'].dt.year):
    df_date_raster_channel.loc[df_temp.index, 'flood_raster_channel'] = np.arange(1, len(df_temp)+1, 1)
#%% Load raster images
# Plot 4 images
channel = 12 # Will plot channel-2, channel-1, channel, channel+1

for channel in range(22, 40):
    # Delete dict_img[key] that key not in range of (channel-2, channel+2)
    [dict_img.pop(key) for key in [key for key in dict_img.keys() if not key in range(channel-2, channel+2)]]
    
    # Load images of selected channels (if not in dict)
    for i in range(channel-2, channel+2):
        if i in dict_img.keys():
            continue
        print(f"Loading channel: {i}")
        img = icedumpy.geo_tools.create_landsat8_color_infrared_img(root_raster, i, 'bgr')
        dict_img[i] = cv2.resize(img, (img.shape[0]//20, img.shape[1]//20))
    
    img = np.vstack((np.hstack((dict_img[channel-2], dict_img[channel-1])),
                     np.hstack((dict_img[channel],   dict_img[channel+1]))))
    
    raster_flood = dict_flood_map[date_raster[channel-1].year]
    img_flood = raster_flood.read(int(df_date_raster_channel.loc[df_date_raster_channel['normal_raster_channel'] == channel, 'flood_raster_channel'].values[0]))
    
    img_flood = np.where(img_flood==1, 255, img_flood)
    img_flood = np.where(img_flood==0, 128, img_flood)
    img_flood = np.where(img_flood==raster_flood.nodata, 0, img_flood)
    img_flood = img_flood.astype('uint8') 
    img_flood = cv2.applyColorMap(img_flood, cv2.COLORMAP_RAINBOW)
    img_flood = cv2.resize(img_flood, (img_flood.shape[0]//10, img_flood.shape[1]//10), interpolation=cv2.INTER_NEAREST)
    
    img = np.hstack((img, img_flood))
    
    cv2.imshow("", img)
    cv2.waitKey(1)