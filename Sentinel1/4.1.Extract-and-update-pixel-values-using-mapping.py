import os
import rasterio
import pandas as pd
from tqdm import tqdm
#%%
sat_type = "S1AB"
root_raster = os.path.join(r"C:\Users\PongporC\Desktop\temp", sat_type.upper())
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
os.makedirs(root_save, exist_ok=True)
#%%
# Get list of files (sort by strip_id)
list_file = os.listdir(root_save)
list_file = sorted(list_file, key=lambda val: val.split(".")[0][-3:])
#%%
# Load for each file (Keep it simply but slowvy. For the sake of PC's memories)
pbar = tqdm(list_file)
for file in pbar:
    pbar.set_description(f"Processing: {file}")
    # Load raster
    strip_id = file.split(".")[0][-3:]
    path_raster = os.path.join(root_raster, f"{sat_type}_{strip_id}.vrt")
    raster = rasterio.open(path_raster)
    
    # Load df 
    path_file = os.path.join(root_save, file)
    df = pd.read_parquet(path_file)
    row = df["row"].values
    col = df["col"].values
    # Loop for each raster's band
    for i in range(2, raster.count):
        # Update only new band
        if not (raster.descriptions[i] in df.columns[7:]):
            img = raster.read(i+1)
            df = df.assign(**{raster.descriptions[i] : img[row, col]})
    
    # Save updated df
    df.to_parquet(path_file)
#%%
