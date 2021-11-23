import os
import datetime
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from rasterio.mask import mask
from Function import create_polygon_from_wkt
#%%
root_df_dri = r"F:\CROP-PIER\CROP-WORK\!DRI-prep\pixel-values"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_temporal"
path_raster_mapping = r"F:\CROP-PIER\CROP-WORK\!DRI-prep\mapping\DRI-mapping-thailand.tif"
raster_mapping = rasterio.open(path_raster_mapping)
columns_ideal = [f"{i}" for i in range(26)]
#%%
# 1: Mapping row, col to each ext_act_id
for file_df_vew in os.listdir(root_df_vew):
    path_save = os.path.join(root_save, f"gistda_dri_temporal_{file_df_vew.split('.')[0][-7:]}.parquet")
    if os.path.exists(path_save):
        continue
    
    print(file_df_vew)
    # Start from 2017
    if int(file_df_vew.split(".")[0][-4:]) < 2017:
        continue
    
    # Load df_dri (all years)
    df_dri = pd.concat([pd.read_parquet(os.path.join(root_df_dri, file)) for file in os.listdir(root_df_dri) if file.split(".")[0][-7:-5] == file_df_vew.split(".")[0][-7:-5]], ignore_index=False)
    df_dri = df_dri.drop(columns=["tambon_pcode", "amphur_cd", "tambon_cd"])
    # Drop duplicates (same [row_col, start_date, dri])
    df_dri = df_dri.reset_index()
    df_dri = df_dri.drop_duplicates(subset=["row_col", "start_date", "dri"])    
    df_dri = df_dri.set_index("row_col")
    df_dri["start_date"] = pd.to_datetime(df_dri["start_date"])
    if not df_dri.index.is_monotonic:
        # If index is not monotonic, sort (increase .loc speed)
        df_dri = df_dri.sort_index()

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

    # 2: Groupby (row, col, plant_date)
    list_df = []
    for (row, col, final_plant_date), df_vew_grp in tqdm(df_vew.groupby(["row", "col", "final_plant_date"])):
        # Select dri from row, col
        try:
            df_dri_grp = df_dri.loc[f"{int(row)}-{int(col)}"].sort_values(by="start_date")
        except:
            continue
        # Filter dri by plant_date (plant_date, plant_date+180)
        df_dri_grp = df_dri_grp.loc[(df_dri_grp["start_date"] >= final_plant_date) & (df_dri_grp["start_date"] <= (final_plant_date+datetime.timedelta(days=180)))]
        if len(df_dri_grp) == 0:
            continue
        
        # Add date_index (t0, t1, ..., tn)
        offset = abs(final_plant_date-df_dri_grp.iloc[0]["start_date"]).days%7 # Keep this 0-6
        df_dri_grp["date_index"] = np.round(((final_plant_date-df_dri_grp["start_date"]).dt.days+offset).abs()/7).astype("int32").astype(str)
        # Append dri data to df_vew
        df_vew_grp[df_dri_grp["date_index"]] = df_dri_grp["dri"]
        list_df.append(df_vew_grp)
        
    df = pd.concat(list_df, ignore_index=True)
    for missing_column in [column for column in columns_ideal if not column in df.columns[34:]]:
        df[missing_column] = np.nan
    df = df[df.columns[:34].tolist()+list(map(str, sorted(df.columns[34:].astype(int))))]
    df = df.rename(columns={i:f"t{i}" for i in df.columns[34:]})
    df.to_parquet(path_save)
#%%