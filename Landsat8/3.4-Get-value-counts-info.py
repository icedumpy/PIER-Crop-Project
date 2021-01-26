import os
import pandas as pd
from tqdm import tqdm
#%%
root_flood = r"F:\CROP-PIER\CROP-WORK\landsat8-dataframe\ls8-flood-value"
root_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8-dataframe\ls8-noflood-value"
root_save = r"F:\CROP-PIER\CROP-WORK\landsat8-dataframe\ls8_value_counts"

columns_data = ['t-2', 't-1', 't+0', 't+1']
#%%
result = pd.DataFrame(columns = ['pathrow', 'p', 'band', 'unique_data', 'occurences'])
for file in tqdm(os.listdir(root_flood)+os.listdir(root_noflood)):
    p = file.split("_")[4][1:]
    pathrow = file.split("_")[5]
    band = file.split("_")[6]
    flood_noflood = file.split("_")[2]
    path_save = os.path.join(root_save, f"ls8_value_counts_{flood_noflood}_p{p}_{pathrow}_{band}.parquet")
    if os.path.exists(path_save):
        continue
    
    if flood_noflood=="flood":
        path_file = os.path.join(root_flood, file)
    else:
        path_file = os.path.join(root_noflood, file)
    df = pd.read_parquet(path_file)
    
    if band=="BQA":
        df = df[(df[columns_data]!=1).all(axis=1)]
        for col in columns_data:
            df[col] = df[col].astype(int)
    else:
        df = df[(df[columns_data]!=0).all(axis=1)]
    
    df['concat'] = df['t-2'].astype(str) + "_" + df['t-1'].astype(str) + "_" + df['t+0'].astype(str) + "_" + df['t+1'].astype(str)
    value_counts = df['concat'].value_counts()
    df_value_count = pd.DataFrame({'pathrow' : pathrow,
                                   'p' : p,
                                   'band' : band,
                                   'unique_data' : value_counts.index,
                                   'occurences': value_counts.values})
    
    df_value_count.to_parquet(path_save)
#%%