import os
import rasterio
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt 
#%%
before = 1
after = 1
img_size = 64
path_raster = r"C:\Users\PongporC\Desktop\temp\S1AB\S1AB_402.vrt"
root_raster_flood = r"G:\!PIER\!Flood-map-groundtruth\Sentinel-1\S1AB\402"
root_save = r"C:\Users\PongporC\Desktop\temp\test"

os.makedirs(os.path.join(root_save, "normal"), exist_ok=True)
os.makedirs(os.path.join(root_save, "flood"), exist_ok=True)
#%%
index_normal = 0
index_flood = 0
raster = rasterio.open(path_raster)
for file in os.listdir(root_raster_flood):
    raster_flood = rasterio.open(os.path.join(root_raster_flood, file))
    
    flood_index = [i+1 for i, band in enumerate(raster.descriptions) if file.split(".")[0].replace("-", "") in band][0]
    if ((flood_index-before) < 1) and (flood_index+after > raster.count):
        continue
    
    img = raster.read(range(flood_index-before, flood_index+after+1))
    img = np.moveaxis(img, 0, -1)
    
    img_flood = raster_flood.read(1)
    
    for row in range(0, img.shape[0]-img_size, img_size):
        for col in range(0, img.shape[1]-img_size, img_size):
            patch = img[row:row+img_size, col:col+img_size]
            patch_flood = (img_flood[row:row+img_size, col:col+img_size] > 0).astype("uint8")
            
            if (patch == 0).all(axis=-1).any():
                continue
            
            if patch_flood.sum() > 0:
                np.save(os.path.join(root_save, "normal", f"{index_normal}_img.npy"), patch)
                np.save(os.path.join(root_save, "normal", f"{index_normal}_mask.npy"), patch_flood)
                index_normal+=1
            else:
                np.save(os.path.join(root_save, "flood", f"{index_flood}_img.npy"), patch)
                np.save(os.path.join(root_save, "flood", f"{index_flood}_mask.npy"), patch_flood)
                index_flood+=1
#%%
