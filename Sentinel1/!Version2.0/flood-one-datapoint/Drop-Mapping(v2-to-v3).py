import os
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
root_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v2"
root_mapping_new = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
#%%
list_strip_id = np.unique([os.path.splitext(file)[0][-3:] for file in os.listdir(root_mapping)]).tolist()
for strip_id in list_strip_id:
    tqdm.write(f"START: {strip_id}")
    list_file = [file for file in os.listdir(root_mapping) if strip_id in file]
    
    raster = gdal.Open(os.path.join(root_raster, f"LSVVS{strip_id}_2017-2019"))
    num_bands = raster.RasterCount 
    width = raster.RasterXSize
    height = raster.RasterYSize    
    valid_mask = np.zeros((height, width), 'uint8')
    for band in range(num_bands):
        raster_band = raster.GetRasterBand(band+1)
        band_name = raster_band.GetDescription()
        
        raster_im = raster_band.ReadAsArray()
        valid_mask += (raster_im>0)

    valid_mask = (valid_mask>(0.8*num_bands)).astype('uint8')
    
    for file in tqdm(list_file):
        if os.path.exists(os.path.join(root_mapping_new, file)):
            continue
        list_df_mapping_new = []
        df_mapping = pd.read_parquet(os.path.join(root_mapping, file))
        
        for new_polygon_id, df_mapping_grp in df_mapping.groupby(['new_polygon_id']):
            row = df_mapping_grp['row'].to_list()
            col = df_mapping_grp['col'].to_list()
            
            if valid_mask[row, col].astype('bool').all():
                df_mapping_new = pd.DataFrame({'strip_id' : int(strip_id),
                                               'row_col' : [(df_mapping_grp['row'].astype(str) + '_' + df_mapping_grp['col'].astype(str)).tolist()]})
                df_mapping_new.index = [new_polygon_id]
                list_df_mapping_new.append(df_mapping_new)
        
        if len(list_df_mapping_new)!=0:
            df_mapping_new = pd.concat(list_df_mapping_new)
            df_mapping_new.to_parquet(os.path.join(root_mapping_new, file))
#%%
