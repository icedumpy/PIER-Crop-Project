import os
import shutil
from tqdm import tqdm
root_src = r"H:\My Drive\!PIER\!landsat8-prep\pixel_cloudmask"
root_dst = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_cloudmask"
years = ['2017', '2018', '2019']
#%% Copy file from Gdrive
for file in tqdm(os.listdir(root_src)):
    if (not "(" in file) and (not ")" in file):
        if os.path.exists(os.path.join(root_dst, file)):
            continue
        if (not "(" in file) and (not ")" in file) and file.endswith(".parquet"):
            if file.split(".")[0][-4:] in years:
                while True:
                    try:
                        tqdm.write(file)
                        shutil.copy(os.path.join(root_src, file), os.path.join(root_dst, file))
                        break
                    except:
                        pass
#%% Re-check all copied files (if the size of loaded file is not the same as the source, re-load again)
for file in tqdm(os.listdir(root_dst)):
    if (not "(" in file) and (not ")" in file):
        #tqdm.write(file)
        if os.path.exists(os.path.join(root_dst, file)):
            if os.path.getsize(os.path.join(root_src, file))!=os.path.getsize(os.path.join(root_dst, file)):
                os.remove(os.path.join(root_dst, file))
                while True:
                    try:
                        tqdm.write(f"Reload {file}")
                        shutil.copy(os.path.join(root_src, file), os.path.join(root_dst, file))
                        break
                    except:
                        pass
#%% Check if can open or not
import pandas as pd
list_broken = []
for file in tqdm(os.listdir(root_dst)):
    if (not "(" in file) and (not ")" in file):
        try:
            df = pd.read_parquet(os.path.join(root_dst, file))
        except:
            list_broken.append(file)
print(list_broken)
#%%