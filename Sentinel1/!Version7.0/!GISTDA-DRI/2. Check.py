import os
import pandas as pd
from tqdm import tqdm
#%%
root_original = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_features = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_features"
#%% Check#1 
total_original = 0
total_features = 0
for file in tqdm(os.listdir(root_features)):
    df_features = pd.read_parquet(os.path.join(root_features, file))
    df_original = pd.read_parquet(os.path.join(root_original, f"vew_plant_info_official_polygon_disaster_all_rice_p{file.split('.')[0][-7:]}.parquet"))
    total_original += len(df_original)
    total_features += len(df_features)
#%% Check#2
columns_all = [f"t{i}" for i in range(26)]
total_features = 0
total_all_nan = 0
for file in tqdm(os.listdir(root_features)):
    df_features = pd.read_parquet(os.path.join(root_features, file))
    total_all_nan += df_features[columns_all].isna().all(axis=1).sum()
    total_features += len(df_features)
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_features\gistda_dri_features_54_2018.parquet")
df.sample(n=1).iloc[0].to_dict()
