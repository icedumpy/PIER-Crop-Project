import os
import datetime
import numpy as np
import rasterio
import rasterio.plot
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
def plot_im(im, ax):
    vmin, vmax = (0, 0.3)
    ax.imshow(im, vmin=vmin, vmax=vmax, cmap='gray')
    ax.set_axis_off()
#%%
root_data = r"F:\CROP-PIER\CROP-WORK\Presentation\20200427"
dict_color = {'TP' : 'blue',
              'TN' : 'green',
              'FP' : 'purple',
              'FN' : 'red'}
#%%
raster_strip_id = 304
year = 2017
#%%
root_data = os.path.join(root_data, str(raster_strip_id))
path_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV\LSVVS{0:d}_2017-2019".format(raster_strip_id)
path_raster_flood_any = os.path.join(root_data, [file for file in os.listdir(root_data) if file.endswith('any.tiff')][0])
path_raster_flood = path_raster_flood_any.replace("_any", "")
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20200427\304\visualize"
if not os.path.exists(root_save):
    os.makedirs(root_save)
#%%
list_polygon = []
for r, _, f in os.walk(root_data):
    for file in f:
        if file.endswith('.shp'):
           list_polygon.append(gpd.read_file(os.path.join(r, file), encoding='CP874'))
polygon = gpd.GeoDataFrame(pd.concat(list_polygon, ignore_index=True))
#%%
polygon['final_plant_date'] = pd.to_datetime(polygon['final_plan'])
polygon['START_DATE'] = pd.to_datetime(polygon['START_DATE'],  errors='coerce')
raster = rasterio.open(path_raster)
raster_flood_any = rasterio.open(path_raster_flood_any)
raster_flood = rasterio.open(path_raster_flood)
list_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster_flood_any.descriptions]
#%%
for i in range(73):
    print(i)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(15, 15))
    raster_im = raster.read(i+1)
    plot_im(raster_im, ax)
    fig.savefig(os.path.join(root_save, f"raster_{str(list_date[i].date())}.png"), dpi=200)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_axis_off()
    rasterio.plot.show((raster_flood, i+1), ax=ax, cmap='gray')
    fig.savefig(os.path.join(root_save, f"flood_{str(list_date[i].date())}.png"), dpi=200)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_axis_off()
    rasterio.plot.show((raster_flood_any, i+1), ax=ax, cmap='gray')
    
    polygon_grp = polygon[(polygon['START_DATE']>list_date[i-1]) & (polygon['START_DATE']<=list_date[i])]
    for result, polygon_to_plot in polygon_grp.groupby(['result']):
        polygon_to_plot.plot(ax=ax, facecolor='none', edgecolor=dict_color[result])
    fig.savefig(os.path.join(root_save, f"flood_any_{str(list_date[i].date())}.png"), dpi=200)
    plt.close('all')
    
    # Load saved image and delete edge
    im = imread(os.path.join(root_save, f"raster_{str(list_date[i].date())}.png"))[:, :, :3]
    im = im[659:2370, 374:2700]
    imsave(os.path.join(root_save, f"raster_{str(list_date[i].date())}.png"), im)
    
    flood_im = imread(os.path.join(root_save, f"flood_{str(list_date[i].date())}.png"))[:, :, :3]
    flood_im = flood_im[659:2370, 374:2700]
    imsave(os.path.join(root_save, f"flood_{str(list_date[i].date())}.png"), flood_im)
    
    flood_im = imread(os.path.join(root_save, f"flood_any_{str(list_date[i].date())}.png"))[:, :, :3]
    flood_im = flood_im[659:2370, 374:2700]
    imsave(os.path.join(root_save, f"flood_any_{str(list_date[i].date())}.png"), flood_im)
#%%
