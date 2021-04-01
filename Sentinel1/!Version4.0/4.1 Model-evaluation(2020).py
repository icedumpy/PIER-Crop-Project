import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_2020"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
#%%
list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
for strip_id in list_strip_id:
    print(strip_id)
    # Load df rice code
    df_rice_code = pd.read_csv(path_rice_code, encoding='cp874')
    df_rice_code = df_rice_code[["BREED_CODE", "photo_sensitive_f"]]
    
    # Load df temporal
    df = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_s{strip_id}.parquet"))
    df = df[df["is_within"]]
    df = df[df["tier"] == 1]
    df.columns = [column[:8] if "_S1" in column else column for column in df.columns]
    
    # Load df vew
    df_vew = pd.concat(
        [pd.read_parquet(os.path.join(root_df_vew, file))
         for file in os.listdir(root_df_vew)
         if file.split(".")[0].split("_")[-1] in df["p_code"].unique().tolist()
         ],
        ignore_index=True
    )
    df_vew = df_vew.loc[(df_vew["DANGER_TYPE_NAME"].isna()) | (df_vew["DANGER_TYPE_NAME"] == "อุทกภัย")]
    df_vew = df_vew[df_vew["ext_act_id"].isin(df["ext_act_id"])]
    df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
    df_vew = pd.merge(df_vew, df_rice_code, on="BREED_CODE", how="inner")
    df_vew = df_vew[["ext_act_id", "final_plant_date", "START_DATE", "loss_ratio", "photo_sensitive_f"]]
    
    # Merge df_vew with df
    df = pd.merge(df, df_vew, how="inner", on="ext_act_id")
    df = df[df.columns[-4:].tolist() + df.columns[:-4].tolist()]
    
    # Get (t0-t29)
    columns = df.columns[11:].tolist()
    columns.insert(0, str((datetime.datetime.strptime(columns[0], "%Y%m%d")-datetime.timedelta(days=6)).date()).replace("-", ""))#%%
    arr = np.zeros((len(df), 30), dtype="float32")
    for index, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        final_plant_date = row["final_plant_date"]
        final_plant_date = str(final_plant_date.date()).replace("-", "")
        index_final_plant_date = [i for i in range(len(columns)-1) if (final_plant_date > columns[i]) and (final_plant_date <= columns[i+1])][0]
        index_final_plant_date += 11
        
        temporal = row[index_final_plant_date:index_final_plant_date+30].values
        if len(temporal) != 30:    
            temporal = np.hstack((temporal, np.zeros((30-len(temporal),))+np.nan))
        arr[index] = temporal
    df = df.iloc[:, :11]
    df = df.assign(**{f"t{i}":arr[:, i] for i in range(30)})
    df = df[df.columns[:3].tolist() + df.columns[4:].tolist() + ["photo_sensitive_f"]]
    
    # Save
    for p_code, df_grp in df.groupby(["p_code"]):
        path_save = os.path.join(root_save , f"df_s1ab_temporal_p{p_code}_s{strip_id}.parquet")
        df_grp.to_parquet(path_save)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
