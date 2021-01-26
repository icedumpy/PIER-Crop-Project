import os
import numpy as np
import pandas as pd
from osgeo import gdal
import icedumpy
from tqdm import tqdm
#%%
# Set initial parameters
root_mapping = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_polygon_id_rowcol_map_prov_scene_merged_v2"
root_raster = r"F:\CROP-PIER\CROP-WORK\LS8_VRT"
root_save = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_pixel_from_mapping_v2"
#%%
#bands = ['B1_TOA', 'B2_TOA', 'B3_TOA', 'B4_TOA', 'B5_TOA', 'B6_TOA', 'BQA']
bands = ['NDVI_TOA']
print(bands)
#%%
# Start loop for each mapping file
for band in bands:
    tqdm.write(f"Start band: {band}")
    for file_mapping in tqdm(os.listdir(root_mapping)[2::3]):

        # Define mapping path
        path_mapping = os.path.join(root_mapping, file_mapping)
        
        # Define save path
        path_save = os.path.join(root_save, f"ls8_pixel_{path_mapping.split('.')[0].split('map')[-1][1:]}_{band}.parquet")
        
        # If save path already exists, skip
        if os.path.exists(path_save):
            tqdm.write(f"{path_save} already exists")
            continue
        
        # Get pathrow of mapping file
        pathrow = os.path.splitext(file_mapping)[0][-6:]
        
        # Try to open raster. If raster is not avaiable or broken, skip
        try:
            raster = gdal.Open(os.path.join(root_raster, pathrow, f"ls8_{pathrow}_{band}.vrt"))
            if raster is None:
                continue
        except:
            continue
        
        # Get date of each raster's band
        raster_date = icedumpy.geo_tools.get_raster_date(raster, 'date', 'gdal')
        raster_date = list(map(str, raster_date))
        
        # Read mapping dataframe
        df_mapping = pd.read_parquet(path_mapping)
        # Reset index (Start from 0 to len(df_mapping)-1)
        df_mapping = df_mapping.reset_index(drop=True)
        
        # Create new dataframe (Modified from mapping dataframe) [1. Drop new_polygon_id] [2. drop duplicated row and col] 
        df_mapping_drop_dup = df_mapping[['row', 'col']].drop_duplicates(subset = ['row', 'col'])
        
        # Reset index of new dataframe
        df_mapping_drop_dup = df_mapping_drop_dup.reset_index(drop=True)
        
        # Get rows and cols
        rows = df_mapping_drop_dup['row'].values
        cols = df_mapping_drop_dup['col'].values
        
        # Loop for each raster band
        # Pre-define array variable for storing pixel values
        raster_values_array = np.zeros((len(df_mapping_drop_dup), raster.RasterCount), dtype='float32')
        for i in range(raster.RasterCount):
            # Get image of each band
            raster_im = raster.GetRasterBand(i+1).ReadAsArray() 
            # Get row, col pixel values and store in raster_values_array
            raster_values_array[:, i] = raster_im[rows, cols]
        
        # Get default columns of mapping dataframe        
        columns = df_mapping_drop_dup.columns
        # Add pixel data values from array to mapping dataframe
        df_mapping_drop_dup = pd.concat([df_mapping_drop_dup, pd.DataFrame(raster_values_array)], axis=1)
        # Add new columns 
        df_mapping_drop_dup.columns = columns.tolist() + raster_date
        
        # Merge row, col pixel values back to main mapping dataframe
        df_mapping = pd.merge(df_mapping, df_mapping_drop_dup, how='left', on=['row', 'col'])
        
        # Check if any nan occurs
        assert ~pd.isna(df_mapping).any().any()
        
        # Save results
        df_mapping.to_parquet(path_save)
#%%