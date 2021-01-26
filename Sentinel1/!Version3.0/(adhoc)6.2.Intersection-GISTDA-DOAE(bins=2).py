import os
import pylab
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rasterio.plot import show
from rasterio.mask import mask
from rasterio import RasterioIOError
from tqdm import tqdm
from icedumpy.df_tools import set_index_for_loc

cmap = matplotlib.cm.get_cmap("Accent", 3)
custom_lines = [
    Patch(facecolor=cmap(1), edgecolor=cmap(1)),
    Patch(facecolor=cmap(2), edgecolor=cmap(2)),
]
#%%
root = r"C:\Users\PongporC\Desktop\Drone Area\gistda_flood_2019\Rasterize"
list_p = list(set([int(file.split("_p")[1][:2]) for file in os.listdir(root) if file.endswith(".tif")]))

gdf_thailand = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-province.shp")
gdf_thailand = set_index_for_loc(gdf_thailand, "ADM1_PCODE")
#%%
pbar = tqdm(list_p)
for p in pbar:
    pbar.set_description(f"Processing: p{p}")
    
    # Load img
    raster_DOAE = rasterio.open(os.path.join(root, f"DOAE_p{p}.tif"))
    img_DOAE, _ = mask(raster_DOAE, [gdf_thailand.loc[f"TH{p}", "geometry"]], indexes=1)
    transform = raster_DOAE.transform
    try:
        img_GISTDA, _ = mask(rasterio.open(os.path.join(root, f"GISTDA_p{p}.tif")), [gdf_thailand.loc[f"TH{p}", "geometry"]], indexes=1)
    except RasterioIOError:
        img_GISTDA = np.zeros_like(img_DOAE)
    
    # Some process 1
    img = np.zeros_like(img_DOAE)
    img = np.where((img_DOAE == 1) & (img_GISTDA == 0), 1, img)
    img = np.where((img_DOAE == 1) & (img_GISTDA == 1), 2, img)
    
    # Some process 2
    TPR = (img == 2).sum()/((img == 1) | (img == 2)).sum()
    
    # Plot
    plt.close("all")
    fig, ax = plt.subplots(figsize = (19.2, 10.8))
    gdf_thailand.loc[f"TH{p}":f"TH{p}"].plot(ax=ax, facecolor="none", edgecolor="black")
    show(img, transform=transform, cmap=cmap, ax=ax, vmin=0, vmax=2)
    ax.set_title(f'{gdf_thailand.loc[f"TH{p}", "ADM1_EN"]}\nTPR: {TPR:.4f}')
    ax.legend(custom_lines, ['(DOAE == 1) & (GISTDA == 0)', '(DOAE == 1) & (GISTDA == 1)'], loc=4)

    # Save plot
    pylab.get_current_fig_manager().window.showMaximized()
    fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20201118\DOAExGISTDA\DOAE", f'{p}_{gdf_thailand.loc[f"TH{p}", "ADM1_EN"]}.png'), 
                dpi=1000)
#%%