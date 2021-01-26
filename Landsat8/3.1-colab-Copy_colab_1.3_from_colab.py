import os
import shutil
from tqdm import tqdm
root_src = r"G:\My Drive\!PIER\ls8_pixel_from_mapping_v2"
root_dst = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_pixel_from_mapping_v2_new"
#%% Copy file from Gdrive 
for file in tqdm(os.listdir(root_src)):
    if (not "(" in file) and (not ")" in file):
        #tqdm.write(file)
        if os.path.exists(os.path.join(root_dst, file)):
            continue
        while True:
            try:
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