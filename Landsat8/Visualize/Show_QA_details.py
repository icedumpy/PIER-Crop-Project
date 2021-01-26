import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
import icedumpy

def cloud(img_QA):
    return (img_QA >> 4) & 1 # Get bit#4

def cloud_confidence(img_QA):
    return (img_QA >> 5) & 3 # Get bit#6:5

def cloud_shadow_confidence(img_QA):
    return (img_QA >> 7) & 3 # Get bit#8:7

def cirrus_confidence(img_QA):
    return (img_QA >> 11) & 3 # Get bit#12:11

def plot_subplot(fig, ax, pos, img, label, colorbar=True):
    row, col = pos
    sub_fig = ax[row, col].imshow(img, cmap='gnuplot')
    ax[row, col].set_title(label)
    ax[row, col].set_xticks([])
    ax[row, col].set_yticks([])
    if colorbar:
        fig.colorbar(sub_fig, ax=ax[row, col])
    return fig, ax

def plot_selected_channel(root_raster, channel):
    img_QA = raster_QA.read(channel)

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    # False color
    fig, ax = plot_subplot(fig, ax, pos=(0, 0),
                           img=icedumpy.geo_tools.create_landsat8_color_infrared_img(root_raster,
                                                                                     channel=channel,
                                                                                     channel_first=False),
                           label="False color",
                           colorbar=False)

    # Cloud
    fig, ax = plot_subplot(fig, ax, pos=(0, 1),
                           img=cloud(img_QA),
                           label="Cloud (1:Yes, 0:No)",
                           colorbar=True)

    # Cloud confidence
    fig, ax = plot_subplot(fig, ax, pos=(0, 2),
                           img=cloud_confidence(img_QA),
                           label="Cloud confidence (3:High, 2:Medium, 1:Low)",
                           colorbar=True)

    # Cloud shadow confidence
    fig, ax = plot_subplot(fig, ax, pos=(1, 1),
                           img=cloud_shadow_confidence(img_QA),
                           label="Cloud shadow confidence (3:High, 2:Medium, 1:Low)",
                           colorbar=True)

    # Cirrus confidence
    fig, ax = plot_subplot(fig, ax, pos=(1, 2),
                           img=cirrus_confidence(img_QA),
                           label="Cirrus confidence (3:High, 2:Medium, 1:Low)",
                           colorbar=True)

    ax[1, 0].axis('off')
    return fig, ax

def main(channel):
    fig, ax = plot_selected_channel(root_raster, channel)
    fig.savefig(os.path.join(root_save, f"{raster_QA.descriptions[channel-1]}.png"), dpi=500, transparent=True, bbox_inches='tight')
    plt.close(fig)
#%%
root_raster = r"G:\!PIER\!FROM_2TB\LS8_VRT\129049"
root_save = r"C:\Users\PongporC\Desktop\QA"
raster_QA = rasterio.open(os.path.join(root_raster, "ls8_129049_BQA.vrt"))
#%%
if __name__=="__main__":
    n_channel = raster_QA.count
    for channel in tqdm(np.arange(1, n_channel+1).tolist()):
        main(channel)
