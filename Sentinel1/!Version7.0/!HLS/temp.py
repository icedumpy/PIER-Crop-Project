import rasterio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
path = r"F:\CROP-PIER\CROP-WORK\!hls-prep\ndvi-16d-noise-reduct-splitted\T48PUC\2019\HLS_V14_NDVI_16D_NOISE_REDUCT_T48PUC_2019-01-05.tif"
raster = rasterio.open(path)
#%%
img = raster.read(1)
#%%
plt.imshow(img, vmin=0, vmax=1, cmap="gray")
#%%
sns.histplot(img[img > 0], bins=100)
