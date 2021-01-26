import os
import shutil
#%%
root_src = r"F:\CROP-PIER\CROP-WORK\Flood-map\one_data_point"
root_dst = r"F:\CROP-PIER\CROP-WORK\Flood-map\nine_data_point"
#%%
for r, d, f in os.walk(root_src):
    for file in f:
        if file.endswith("mask.tif"):
            path_src = os.path.join(r, file)
            path_dst = path_src.replace("one_", "nine_")
            
            os.makedirs(os.path.dirname(path_dst), exist_ok = True)
            
            shutil.copyfile(path_src, path_dst)
