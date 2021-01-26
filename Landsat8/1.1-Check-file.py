import os
import datetime
import pandas as pd
import rasterio
from tqdm import tqdm
#%%
root_raster = r"F:\CROP-PIER\CROP-WORK\LS8"
ignored_file_format = ['MTL.txt', '.xml', '.vrt']
#%% Check raster files
# Loop over every file in the root_raster folder to get the total raster files
total = 0
for root, dirs, files in os.walk(root_raster):
    total+=1

# Get file size
tqdm.write("Check raster files")
df_raster_size = pd.DataFrame(columns=['filepath', 'filename', 'band', 'size(MB)'])
for root, dirs, files in tqdm(os.walk(root_raster), total=total):
    for file in files:
        if file.endswith(tuple(ignored_file_format)):
            continue
        else:
            path_file = os.path.join(root, file)
            if file.endswith("BQA.TIF"):
                band = "BQA"
            else:
                band = file.split(".")[0].split("_")[-2]

            df_raster_size = df_raster_size.append(pd.Series({ 'filepath' : path_file,
                                                                                     'filename' : file,
                                                                                     'band' : band,
                                                                                     'size(MB)' : os.path.getsize(path_file)/(1024**2)
                                                                                    }),
                                                                    ignore_index = True
                                                                    )

# Get list of file that file size is less then quantile xxth of each band
list_check_raster = []
for band, df_raster_size_grp in df_raster_size.groupby(['band']):
    quantile_value = df_raster_size_grp['size(MB)'].quantile(0.025)
    list_check_raster+=df_raster_size_grp[df_raster_size_grp['size(MB)'] < quantile_value]['filepath'].to_list()    

# Check if file is defect or not
list_defect = []
for path_file in tqdm(list_check_raster):
    try:
        with rasterio.open(path_file) as raster:
            if raster is None:
                list_defect.append(path_file)
                tqdm.write("None raster file")
                raise Exception("Can open but None")
            elif (raster.read(1) == 0).all():
                list_defect.append(path_file)
                tqdm.write("All zero image file")
                raise Exception("Can open and read but all zero")
    except:
        list_defect.append(path_file)
if len(list_defect)==0:
    print("All pass")
else:
    print("Defect alert")
    print(list_defect)
#%% Every folder should has 8 files
for r, d, f in os.walk(root):
    if len(f)!=8  and len(f)!=0:
        print(r)
#%% Get raster details
# Get pathrow, date, band, filepath
tqdm.write("Get raster details")
df_pathrow_date_filepath = pd.DataFrame(columns = ['pathrow', 'date', 'band', 'filepath'])
for root, dirs, files in tqdm(os.walk(root_raster), total=total):
    for file in files:
        if file.endswith(tuple(ignored_file_format)):
            continue
        else:
            path_file = os.path.join(root_raster, file)
            if file.endswith("BQA.TIF"):
                band = "BQA"
            else:
                band = file.split(".")[0].split("_")[-2]
            pathrow, date = os.path.basename(r).split("_")[1:]
            date = datetime.datetime.strptime(date, "%Y%m%d").date()
            df_pathrow_date_filepath = df_pathrow_date_filepath.append(pd.Series({'pathrow':pathrow, 
                                                                                 'date':date,
                                                                                 'band':band,
                                                                                 'filepath':os.path.join(r, file)}), 
                                                                 ignore_index=True)
#%%
# Check datetime with ideal datetime
df_missing_file = pd.DataFrame(columns=['pathrow', 'date'])
for pathrow, df_pathrow_date_filepath_grp in df_pathrow_date_filepath.groupby(['pathrow']):
    ls8_date = pd.unique(df_pathrow_date_filepath_grp['date'])
    df_date_ideal = pd.DataFrame(pd.date_range(start=ls8_date[0], end=ls8_date[-1], freq='16D').date, columns=['ideal'])
    df_date_ideal['available'] = df_date_ideal['ideal'].isin(ls8_date)
    if (~df_date_ideal['available']).any():
        df_missing_file = df_missing_file.append({'pathrow':pathrow, 'date':df_date_ideal[~df_date_ideal['available']]['ideal'].tolist()}, ignore_index=True)
if len(df_missing_file)==0:
    tqdm.write("All pass")
else:
    print(df_missing_file)