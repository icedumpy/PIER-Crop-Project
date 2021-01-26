import os
import shutil
from osgeo import gdal
from tqdm import tqdm
#%%
def check_raster(path):
    try:
        raster = gdal.Open(path)
        if raster is None:
            return False
    except:
        return False
    return True        
#%%
#root_src = r"G:\!PIER\!FROM_2TB\Complete_VV_separate\S1A"
#root_dst = r"H:\My Drive\!PIER\!sen1-prep\processed_data_warp\S1A"

root_src = r"H:\My Drive\!PIER\!sen1-prep\processed_data_warp\S1A"
root_dst = r"G:\!PIER\!FROM_2TB\Complete_VV_separate\S1A"
#%%
list_ignored = ["raw", "ipynb_checkpoints", ".ipynb_checkpoints", "desktop.ini"]
list_folder = [folder for folder in os.listdir(root_src) if folder not in list_ignored]
#%% Download
for folder in tqdm(list_folder):
    path_folder_src = os.path.join(root_src, folder)
    path_folder_dst = os.path.join(root_dst, folder)
    
    if not os.path.exists(path_folder_dst):
        os.mkdir(path_folder_dst)
    
    for file in os.listdir(path_folder_src):
        if file in list_ignored:
            continue
        
        path_file_src = os.path.join(path_folder_src, file)
        if "(" in file:
            path_file_dst = os.path.join(path_folder_dst, ".".join([file.split(" ")[0], file.split(".")[1]]))
        else:
            path_file_dst = os.path.join(path_folder_dst, file)
            
        tqdm.write(f"Start copying: {file}")
        
        if os.path.exists(path_file_dst):
            tqdm.write(f"File already exists: {file}")
            tqdm.write("")
            continue
        
        while True:
            try:
                shutil.copy(path_file_src, path_file_dst)
                assert os.path.exists(path_file_dst)
                if check_raster(path_file_dst):
                    tqdm.write(f"Finish copying: {file}")
                    tqdm.write("")
                    break
            except:
                pass
#%%
# Remove dup files
#for root, _, files in os.walk(root_dst):
#    for file in files:
#        if ("(" in file) and (")" in file):
#            print(file)
#            try:
#                rename = ".".join([file.split(" ")[0], file.split(".")[1]])
#                os.rename(os.path.join(root, file), os.path.join(root, rename))
#            except FileExistsError:
#                os.remove(os.path.join(root, file))
#%%
