import os
import datetime
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
#%%
root_raster = r"G:\!PIER\!FROM_2TB\LS8"
root_ref = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_pixel_rowcol_map"
root_save = r"G:\!PIER\!FROM_2TB\LS8_VRT"
ignored_file_format = ['MTL.txt', '.vrt', 'BQA.tif', '.xml', 'NDVI_TOA.TIF', 'B7_TOA.TIF', 'B9_TOA.TIF']
#%% Get Info of every raw raster (pathrow, date, band, filepath)
total = 0
for root, dirs, files in tqdm(os.walk(root_raster)):
    total+=1

df_pathrow_date_filepath = pd.DataFrame(columns = ['pathrow', 'date', 'band', 'filepath'])
for root, dirs, files in tqdm(os.walk(root_raster), total=total):
    for file in files:
        # Skip QA and MTL file
        if file.endswith(tuple(ignored_file_format)):
            continue
        else:
            if "TOA" in file:
                band = "_".join(file.split(".")[0].split("_")[-2:])
            else:
                band = file.split(".")[0].split("_")[-1]
            pathrow, date = os.path.basename(root).split("_")[1:]
            date = datetime.datetime.strptime(date, "%Y%m%d").date()
            df_pathrow_date_filepath = df_pathrow_date_filepath.append(pd.Series({'pathrow':pathrow, 
                                                                                  'date':date,
                                                                                  'band':band,
                                                                                  'filepath':os.path.join(root, file)}),
                                                                 ignore_index=True)
#%% Create virtual raster of each pathrow and band
for pathrow in tqdm(pd.unique(df_pathrow_date_filepath['pathrow'])):
    raster_ref = gdal.Open(os.path.join(root_ref, f"ls8_pixel_rowcol_map_{pathrow}.tif"))
    for band in tqdm(pd.unique(df_pathrow_date_filepath['band'])):
        
        outvrt = os.path.join(root_save, pathrow, f'ls8_{pathrow}_{band}.vrt')
        os.makedirs(os.path.dirname(outvrt), exist_ok=True)
        
        df_pathrow_date_filepath_selected = df_pathrow_date_filepath[(df_pathrow_date_filepath['pathrow']==pathrow) & (df_pathrow_date_filepath['band']==band)]
        df_pathrow_date_filepath_selected = df_pathrow_date_filepath_selected.sort_values(by=['date'])
        
        options = gdal.BuildVRTOptions(
                                        # outputBounds=(x_min, ymin, x_max, y_max)
                                        outputBounds=(raster_ref.GetGeoTransform()[0], raster_ref.GetGeoTransform()[3] + raster_ref.RasterYSize*raster_ref.GetGeoTransform()[5], raster_ref.GetGeoTransform()[0] + raster_ref.RasterXSize*raster_ref.GetGeoTransform()[1], raster_ref.GetGeoTransform()[3]),
                                        resampleAlg = 'near',
                                        separate=True,
                                        srcNodata = 0,
                                        VRTNodata = 0
                                      )
        
        outds = gdal.BuildVRT(outvrt, df_pathrow_date_filepath_selected['filepath'].tolist(), options=options)
        for idx, date in enumerate(df_pathrow_date_filepath_selected['date']):
            outds.GetRasterBand(idx+1).SetDescription(str(date))
        outds.FlushCache()
        del outds

