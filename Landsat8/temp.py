import os
import numpy as np
import rasterio
from icedumpy.ls8_tools import LS8_QA
from icedumpy.geo_tools import create_tiff, get_raster_date
#%%
def load_raster_img_and_shadow_mask(dict_raster, channel):
    # Load raster images
    # Load cloud shadow confidence (high)
    list_raster_img = []
    for band in dict_raster.keys():
        if band == "BQA":
            mask_shadow = (LS8_QA(dict_raster[band], channel=channel)).cloud_shadow_confidence
            mask_shadow = mask_shadow==3
        else:
            list_raster_img.append(dict_raster[band].read([channel]))

    raster_imgs = np.vstack(list_raster_img)
    return raster_imgs, mask_shadow

def main(folder_raster, folder_save, bands):

    # Load all of the selected raster bands into dict (dict_raster)
    dict_raster = dict()
    for file in os.listdir(folder_raster):
        file_band = os.path.splitext(file)[0].split("_")[2]
        if file_band in bands:
            dict_raster[file_band] = rasterio.open(os.path.join(folder_raster, file))
    # Check correctness of dict_raster
    assert len(set(map(lambda val: val.count, dict_raster.values()))) == 1

    # Get raster information (date, width, height)
    date_raster = get_raster_date(dict_raster[file_band], datetimetype="datetime")
    width = list(set(map(lambda val: val.width , dict_raster.values())))[0]
    heigth = list(set(map(lambda val: val.height, dict_raster.values())))[0]

    # Loop for each year
    for year in np.unique(list(map(lambda val: val.year, date_raster))):
        print("START Year:", year)

        # Create numpy array for index raster (tell which index is min)
        img_indices  = np.zeros((heigth, width), dtype=np.uint8)

        # Get list of channel of this loop's year ex. [1, 2, ..., 22], [23, 24, ..., 45]
        list_channel = np.where(list(map(lambda val:val.year==year, date_raster)))[0] + 1
        for loop_idx, channel in enumerate(list_channel):
            print("Process channel:", channel)

            # Load raster and shadow mask of selected channel
            raster_imgs, mask_shadow = load_raster_img_and_shadow_mask(dict_raster, int(channel))

            # For first loop of selected year
            if loop_idx == 0:
                # Copy raster image of the first channel of selected year
                img_min = raster_imgs.copy()
                # If shadow mask, change value to 1 (to the maximum) becaues we want to ignore cloud shadow
                # The next raster's channel with handle it (should be less than 1)
                rows, cols = np.where(mask_shadow)
                img_min[:, rows, cols] = 1

            else:
                # For others loop

                # Find row, col where new raster image pixel value is less then the "img_min" -(1)
                # Ignore shadow mask(Get only not shadow mask) - (2)
                # AND Operation ((1), (2))
                rows, cols = np.where(np.logical_and(
                    np.logical_and.reduce(img_min > raster_imgs, axis=0),
                    ~mask_shadow)
                )

                # Update value the less than "img_min" and not shadow mask
                img_min[:, rows, cols] = raster_imgs[:, rows, cols]
                # Update index (which channel-1 is min at that specific pixel(row, col))
                img_indices[rows, cols] = loop_idx+1

        # Reverse first channel (B3, B4, B5) >> (B5, B4, B3)
        img_min = img_min[::-1 ,:, :]

        # Save "img_min"
        create_tiff(path_save = os.path.join(folder_save, f"{year}_min"),
                    im = img_min,
                    projection = dict_raster[bands[0]].crs.to_wkt(),
                    geotransform = dict_raster[bands[0]].get_transform(),
                    drivername = "ENVI",
                    list_band_name = bands[::-1][1:],
                    channel_first = True,
                    )

        # Save "indices"
        create_tiff(path_save = os.path.join(folder_save, f"{year}_index"),
                    im = np.expand_dims(img_indices, axis=0),
                    projection = dict_raster[bands[0]].crs.to_wkt(),
                    geotransform = dict_raster[bands[0]].get_transform(),
                    drivername = "ENVI",
                    channel_first = True,
                    dtype = "uint8"
                    )
        print()
        # Bye bye year xxxx

if __name__=='__main__':
    root_raster = r"G:\!PIER\!FROM_2TB\LS8_VRT"
    bands = ["B3", "B4", "B5", "BQA"]
    for folder in os.listdir(root_raster):
        folder_raster = rf"G:\!PIER\!FROM_2TB\LS8_VRT\{folder}"
        folder_save = folder_raster.replace("LS8_VRT", "LS8_VRT_MIN")
        os.makedirs(folder_save, exist_ok=True)

        print("START PATHROW:", folder)
        main(folder_raster, folder_save, bands)
#%%