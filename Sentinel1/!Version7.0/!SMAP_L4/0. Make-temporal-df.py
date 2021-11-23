import os
import datetime
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from rasterio.mask import mask
from Function import create_polygon_from_wkt
#%%
def get_temporal_df(df_vew, df_sm, column_sm):
    df_sm = df_sm.drop(columns=["tambon_pcode", "amphur_cd", "tambon_cd"])
    
    # Drop duplicates
    df_sm = df_sm.reset_index()
    df_sm = df_sm.drop_duplicates(subset=["row_col", "start_date", column_sm])
    
    # Find mean (day-level)
    df_sm["start_date"] = df_sm["start_date"].str.slice(0, 10)
    df_sm = df_sm.groupby(["row_col", "start_date"]).agg("mean").reset_index()
    df_sm = df_sm.set_index("row_col")
    df_sm["start_date"] = pd.to_datetime(df_sm["start_date"])
    if not df_sm.index.is_monotonic:
        # If index is not monotonic, sort (increase .loc speed)
        df_sm = df_sm.sort_index()
        
    # 2: Groupby (row, col, plant_date)
    list_df = []
    for (row, col, final_plant_date), df_vew_grp in tqdm(df_vew.groupby(["row", "col", "final_plant_date"])):
        # Select dri from row, col
        try:
            df_sm_grp = df_sm.loc[f"{int(row)}-{int(col)}"].sort_values(by="start_date")
        except:
            continue
        # Filter dri by plant_date (plant_date, plant_date+180)
        df_sm_grp = df_sm_grp.loc[(df_sm_grp["start_date"] >= final_plant_date) & (df_sm_grp["start_date"] <= (final_plant_date+datetime.timedelta(days=180)))]
        if len(df_sm_grp) == 0:
            continue
        
        # Add date_index (t0, t1, ..., tn)
        offset = abs(final_plant_date-df_sm_grp.iloc[0]["start_date"]).days%1 # Keep this 0-1
        df_sm_grp["date_index"] = np.round(((final_plant_date-df_sm_grp["start_date"]).dt.days+offset).abs()/1).astype("int32").astype(str)
        # Append dri data to df_vew
        df_vew_grp[df_sm_grp["date_index"]] = df_sm_grp[column_sm]
        list_df.append(df_vew_grp)
        
    df = pd.concat(list_df, ignore_index=True)
    for missing_column in [column for column in columns_ideal if not column in df.columns[34:]]:
        df[missing_column] = np.nan
    df = df[df.columns[:34].tolist()+list(map(str, sorted(df.columns[34:].astype(int))))]
    df = df.rename(columns={i:f"t{i}" for i in df.columns[34:]})
    return df
#%%
root_df_smap = r"F:\CROP-PIER\CROP-WORK\!SMAP_L4-prep\pixel-values"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_save_smsurface = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smsurface_temporal"
root_save_smrootzone = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smrootzone_temporal"
path_raster_mapping = r"F:\CROP-PIER\CROP-WORK\!SMAP_L4-prep\mapping\SMAP-L4-mapping-thailand.tif"
raster_mapping = rasterio.open(path_raster_mapping)
columns_ideal = [f"{i}" for i in range(181)]
#%%
# 1: Mapping row, col to each ext_act_id
# file_df_vew = "vew_plant_info_official_polygon_disaster_all_rice_p30_2015.parquet"
for file_df_vew in os.listdir(root_df_vew)[3::4]:
    print(file_df_vew)
    path_save_smrootzone = os.path.join(root_save_smrootzone, f"SMAP_L4_smrootzone_temporal_{file_df_vew.split('.')[0][-7:]}.parquet")
    path_save_smsurface = os.path.join(root_save_smsurface, f"SMAP_L4_smsurface_temporal_{file_df_vew.split('.')[0][-7:]}.parquet")
    if os.path.exists(path_save_smrootzone) and os.path.exists(path_save_smsurface):
        continue
    
    # Load df_vew
    df_vew = pd.read_parquet(os.path.join(root_df_vew, file_df_vew))
    list_row = []
    list_col = []
    for index, serie_vew in df_vew.iterrows():
        polygon = create_polygon_from_wkt(serie_vew["polygon"])
        try:
            row_col, _ = mask(raster_mapping, [polygon.centroid], crop=True, all_touched=False, nodata=-999)
            row_col = row_col.reshape(-1)
            row, col = row_col
        except ValueError:
            row = col = np.nan
        list_row.append(row)
        list_col.append(col)
    df_vew["row"] = list_row
    df_vew["col"] = list_col
    # Drop nan (row, col)
    df_vew = df_vew.dropna(subset=["row", "col"])
    
    df_smrootzone = pd.concat([pd.read_parquet(os.path.join(root_df_smap, file)) for file in os.listdir(root_df_smap) if file.split(".")[0][-7:-5] == file_df_vew.split(".")[0][-7:-5] and "smrootzone" in file], ignore_index=False)
    df = get_temporal_df(df_vew, df_smrootzone, "sm_rootzone")
    df.to_parquet(path_save_smrootzone)
    del df, df_smrootzone
    
    df_smsurface = pd.concat([pd.read_parquet(os.path.join(root_df_smap, file)) for file in os.listdir(root_df_smap) if file.split(".")[0][-7:-5] == file_df_vew.split(".")[0][-7:-5] and "smsurface" in file], ignore_index=False)
    df = get_temporal_df(df_vew, df_smsurface, "sm_surface")
    df.to_parquet(path_save_smsurface)
    del df, df_smsurface
#%%

#%%