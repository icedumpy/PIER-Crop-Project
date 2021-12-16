import os
import pandas as pd
from tqdm import tqdm
#%%
df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
df = df.reset_index(drop=True)
df.head()
#%%
pbar = tqdm(df.prov_cd.unique())
list_df_vew = []
for p in pbar:
    pbar.set_description("Processing P:%s"%(p))
    df_vew = pd.concat([
        pd.read_parquet(rf"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year\vew_plant_info_official_polygon_disaster_all_rice_p{p}_2018.parquet"),
        pd.read_parquet(rf"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year\vew_plant_info_official_polygon_disaster_all_rice_p{p}_2019.parquet"),
        pd.read_parquet(rf"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year\vew_plant_info_official_polygon_disaster_all_rice_p{p}_2020.parquet")
    ])
    df_vew = df_vew[["ext_act_id", "polygon"]]
    df_vew = df_vew[df_vew["ext_act_id"].isin(df["ext_act_id"])]
    list_df_vew.append(df_vew)
    # df = pd.merge(left=df, right=df_vew, how="left", on="ext_act_id")
#%%
df_vew = pd.concat(list_df_vew, ignore_index=True)
del list_df_vew
df = pd.merge(left=df, right=df_vew, how="left", on="ext_act_id")
df.to_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
#%%