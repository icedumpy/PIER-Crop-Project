import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping, set_index_for_loc
tqdm.pandas()
#%%
def add_anchor_date(df_vew):
    list_anchor_date = []
    for vew in tqdm(df_vew.itertuples(), total=len(df_vew)):
        if vew.DANGER_TYPE_NAME == None: # No flood
            anchor_date = vew.final_plant_date + datetime.timedelta(days=np.random.randint(1, 181))
        else:
            anchor_date = vew.START_DATE
        list_anchor_date.append(anchor_date)
    df_vew = df_vew.assign(ANCHOR_DATE = list_anchor_date)
    return df_vew

def get_column_index_from_date(df_content, date):
    column = df_content.loc[((df_content["start"] < date) & (df_content["stop"] >= date))].index[0]
    return column

def get_column_index(columns_s1_temporal, date):
    for i, column_s1_temporal in enumerate(columns_s1_temporal):
        if column_s1_temporal >= date:
            if i != 0:
                return i-1
            else:
                raise Exception('Out of Index')
    raise Exception('Out of Index')
    
def add_missing_columns(df_s1_temporal):
    columns_s1_temporal_ideal = pd.date_range(df_s1_temporal.columns[0], df_s1_temporal.columns[-1], freq="6D")
    for column in columns_s1_temporal_ideal:
        column = str(column.date()).replace("-", "")
        if not column in df_s1_temporal.columns:
            df_s1_temporal = df_s1_temporal.assign(**{column:np.nan})
    df_s1_temporal = df_s1_temporal.reindex(sorted(df_s1_temporal.columns), axis=1)
    return df_s1_temporal
#%%
root_df_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
os.makedirs(root_df_temporal, exist_ok=True)
#%%
strip_id = "402"
#%%
# Load df mapping
df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
#%%
for p in list_p[0::2]:
    print(strip_id, p)
    path_save = os.path.join(root_df_temporal, f"df_s1ab_temporal_p{p}_s{strip_id}.parquet")
    if os.path.exists(path_save):
        continue
    
    # Load df vew of each p
    df_vew = load_vew(root_df_vew, list_p=[p])
    df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
    df_vew = pd.merge(df_vew, df_mapping, how="inner", on=["new_polygon_id"])
    if len(df_vew) == 0:
        continue

    # load df s1_temporal (S1AB backscattering coef(s) starting from 2018-06-01 till the end of 2020)
    df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_p{p}_s{strip_id}.parquet"))
    df_s1_temporal = df_s1_temporal.loc[df_s1_temporal.new_polygon_id.isin(df_mapping.new_polygon_id.unique())]
    df_s1_temporal = set_index_for_loc(df_s1_temporal, column="new_polygon_id")
    df_s1_temporal = df_s1_temporal.iloc[:, 7:]
    df_s1_temporal.columns = [column[:8] for column in df_s1_temporal.columns]
    df_s1_temporal = add_missing_columns(df_s1_temporal)
    columns_s1_temporal = np.array([datetime.datetime.strptime(column, "%Y%m%d") for column in df_s1_temporal.columns])
    columns_s1_temporal = np.insert(columns_s1_temporal, 0, columns_s1_temporal[0]-datetime.timedelta(days=6))
    
    # Filter df_vew by final_plant_date (later or equal temporal first date)
    df_vew = df_vew.loc[df_vew["final_plant_date"] >= datetime.datetime.strptime(df_s1_temporal.columns[0], "%Y%m%d")] # This is sentinal1(A|B) first date
    
    # Change harvest date to plant date + 180 days
    df_vew["final_harvest_date"] = df_vew["final_plant_date"] + datetime.timedelta(days=180)
    
    # Want to get only the data that within the crop cycle (plantdate to plantdate+180days)
    list_df = []
    for vew in tqdm(df_vew.itertuples(), total=len(df_vew)):
        # Extract useful infomation
        try:
            new_polygon_id = vew.new_polygon_id
            date_plant = vew.final_plant_date
            date_harvest = vew.final_harvest_date
            # Get column index
            column_plant = get_column_index(columns_s1_temporal, date_plant)
            column_harvest = get_column_index(columns_s1_temporal, date_harvest)
            
            # Get pixel data (plant -> harvest) # 31 periods
            arr_s1_temporal = df_s1_temporal.loc[new_polygon_id].values
            if len(arr_s1_temporal.shape) == 1:
                arr_s1_temporal = arr_s1_temporal.reshape(1, -1)
        
            arr_s1_temporal = arr_s1_temporal[:, column_plant:column_harvest+1]
            assert arr_s1_temporal.shape[-1] == 31
            
            df = pd.DataFrame(arr_s1_temporal, columns=[f"t{i}" for i in range(arr_s1_temporal.shape[-1])])
            df["new_polygon_id"] = new_polygon_id
            df["PLANT_PROVINCE_CODE"] = vew.PLANT_PROVINCE_CODE
            df["PLANT_AMPHUR_CODE"] = vew.PLANT_AMPHUR_CODE
            df["PLANT_TAMBON_CODE"] = vew.PLANT_TAMBON_CODE
            df["ext_act_id"] = vew.ext_act_id
            df["BREED_CODE"] = vew.BREED_CODE
            df["final_plant_date"] = date_plant
            df["final_harvest_date"] = date_harvest
            df["START_DATE"] = vew.START_DATE
            df["loss_ratio"] = vew.loss_ratio
            df["final_polygon"] = vew.final_polygon
            df["polygon_area_in_square_m"] = vew.polygon_area_in_square_m
            df["row"] = vew.row
            df["col"] = vew.col
            list_df.append(df)
        except:
            pass
    
    df = pd.concat(list_df, ignore_index=True)
    df.to_parquet(path_save)






