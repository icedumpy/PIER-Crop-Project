import os
import pandas as pd
#%%
root_df_temporal =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_temporal"
root_df_dri = r"F:\CROP-PIER\CROP-WORK\!DRI-prep\pixel-values"
#%%
for file in os.listdir(root_df_temporal):
    df_temporal = pd.read_parquet(os.path.join(root_df_temporal, file))
    df_dri = pd.read_parquet(os.path.join(root_df_dri, f"df_dri_features_{file.split('.')[0][-7:]}.parquet"))
    df_dri = df_dri.drop(columns=["tambon_pcode", "amphur_cd", "tambon_cd"])
    df_dri = df_dri.reset_index()
    df_dri = df_dri.drop_duplicates(subset=["row_col", "start_date", "dri"])    
    df_dri = df_dri.set_index("row_col")
    df_dri = df_dri.sort_index()
    break
#%%
for index, serie_vew in df_temporal.iterrows():
    df_dri_grp = df_dri.loc[f"{serie_vew['row']}-{serie_vew['col']}"].sort_values(by="start_date")
    
    break
    pass
#%%



















