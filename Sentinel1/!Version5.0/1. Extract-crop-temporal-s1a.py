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
            anchor_date = vew.DANGER_DATE
        list_anchor_date.append(anchor_date)
    df_vew = df_vew.assign(ANCHOR_DATE = list_anchor_date)
    return df_vew

def get_column_index(columns_s1_temporal, date):
    for i, column_s1_temporal in enumerate(columns_s1_temporal):
        if column_s1_temporal >= date:
            if i != 0:
                return i-1
            else:
                raise Exception('Out of Index')
    raise Exception('Out of Index')

def add_missing_columns(df_s1_overall):
    df_s1_overall = df_s1_overall.copy()
    columns_s1_temporal_ideal = pd.date_range(df_s1_overall.columns[7], df_s1_overall.columns[-1], freq="12D")
    for column in columns_s1_temporal_ideal:
        column = str(column.date()).replace("-", "")
        if not column in df_s1_overall.columns[7:]:
            df_s1_overall = df_s1_overall.assign(**{column:np.nan})
            
    df_s1_overall = df_s1_overall[df_s1_overall.columns[:7].tolist()+sorted(df_s1_overall.columns[7:])]
    return df_s1_overall
#%%
root_df_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1a_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_vew_plant_info_official_polygon_disaster_all_rice_by_year_mapping(at-False)"
root_df_s1_overall = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1a_vew_plant_info_official_polygon_disaster_all_rice_by_year_pixel(at-False)"
os.makedirs(root_df_temporal, exist_ok=True)
#%%
# Set dtypes
dict_dtypes = {
    't0': "float32",
    't1': "float32",
    't2': "float32",
    't3': "float32",
    't4': "float32",
    't5': "float32",
    't6': "float32",
    't7': "float32",
    't8': "float32",
    't9': "float32",
    't10': "float32",
    't11': "float32",
    't12': "float32",
    't13': "float32",
    't14': "float32",
    'EXT_PROFILE_CENTER_ID': "int64",
    'PLANT_PROVINCE_CODE': "int32",
    'PLANT_AMPHUR_CODE': "int32",
    'PLANT_TAMBON_CODE': "int32",
    'ext_act_id': "int64",
    'TYPE_CODE': "int32",
    'BREED_CODE': "int32",
    'ACT_RAI_ORI': "int32",
    'ACT_NGAN_ORI': "int32",
    'ACT_WA_ORI': "int32",
    'TOTAL_ACTUAL_PLANT_AREA_IN_WA': "int32",
    'in_season_rice_f': "int32",
    'loss_ratio': 'float32',
    'polygon_area_in_square_m': 'float32',
    'row': "int32",
    'col': "int32"
}
#%%
list_strip_id = ["101", "102", "103", "104", "105", "106", "107", "108", "109", 
                 "201", "202", "203", "204", "205", "206", "207", "208",
                 "301"]
for strip_id in list_strip_id:
    # Load df mapping
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
    list_p = df_mapping["p_code"].unique().tolist()
    
    for p in list_p:
        print(strip_id, p)
        path_save = os.path.join(root_df_temporal, f"df_s1a_temporal_p{p}_s{strip_id}.parquet")
        if os.path.exists(path_save):
            continue

        # Load df vew of each p
        df_vew = pd.concat([pd.read_parquet(os.path.join(root_df_vew, file)) for file in os.listdir(root_df_vew) if file.split(".")[0].split("_")[-2][1:] in [p]], ignore_index=True)

        df_s1_overall = pd.read_parquet(os.path.join(root_df_s1_overall, f"df_s1a_pixel_p{p}_s{strip_id}.parquet"))
        df_s1_overall = df_s1_overall.loc[df_s1_overall.ext_act_id.isin(df_mapping.ext_act_id.unique())]
        df_s1_overall = set_index_for_loc(df_s1_overall, column="ext_act_id")
        df_s1_overall.columns = [column if not "S1" in column else column[:8] for column in df_s1_overall.columns]
        df_s1_overall = add_missing_columns(df_s1_overall)
        columns_s1_temporal = np.array([datetime.datetime.strptime(column, "%Y%m%d") for column in df_s1_overall.columns[7:]])
        columns_s1_temporal = np.insert(columns_s1_temporal, 0, columns_s1_temporal[0]-datetime.timedelta(days=12))
        
        # Filter df_vew by final_plant_date (later or equal temporal first date)
        df_vew = df_vew.loc[df_vew["final_plant_date"] >= datetime.datetime.strptime(df_s1_overall.columns[7], "%Y%m%d")] # This is sentinal1(A|B) first date

        # Change harvest date to plant date + 180 days (change to 210 (t35))
        df_vew["final_harvest_date"] = df_vew["final_plant_date"] + datetime.timedelta(days=180)
        
        # Add loss ratio
        df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
        
        # Drop deflect
        df_vew = df_vew[~df_vew["BREED_CODE"].isna()]
        df_vew = df_vew[~df_vew["TOTAL_ACTUAL_PLANT_AREA_IN_WA"].isna()]
        
        # Want to get only the data that within the crop cycle (plantdate to plantdate+210 days)
        list_df = []
        for vew in tqdm(df_vew.itertuples(), total=len(df_vew)):
            try:
                ext_act_id = vew.ext_act_id
                date_plant = vew.final_plant_date
                date_harvest = vew.final_harvest_date
                # Get column index
                column_plant = get_column_index(columns_s1_temporal, date_plant)
                column_harvest = get_column_index(columns_s1_temporal, date_harvest)

                # Get pixel data (plant -> harvest) 
                arr_s1_overall = df_s1_overall.loc[ext_act_id]

                arr_s1_temporal = arr_s1_overall[df_s1_overall.columns[7:]].values
                if len(arr_s1_temporal.shape) == 1:
                    arr_s1_temporal = arr_s1_temporal.reshape(1, -1)

                arr_s1_temporal = arr_s1_temporal[:, column_plant:column_harvest]
                assert arr_s1_temporal.shape[-1] == 15

                df = pd.DataFrame(arr_s1_temporal, columns=[f"t{i}" for i in range(arr_s1_temporal.shape[-1])])
                df = df.assign(**{df_vew.columns[i]: vew[i+1] for i in range(len(df_vew.columns))})
                
                try:
                    df["polygon_area_in_square_m"] = arr_s1_overall["polygon_area_in_square_m"].values
                    df["row"] = arr_s1_overall["row"].values
                    df["col"] = arr_s1_overall["col"].values
                except AttributeError: # Case 1-D
                    df["polygon_area_in_square_m"] = arr_s1_overall["polygon_area_in_square_m"]
                    df["row"] = arr_s1_overall["row"]
                    df["col"] = arr_s1_overall["col"]
                list_df.append(df)
            except: 
                # print(ext_act_id)
                pass
        
        # Skip if empty
        if len(list_df) == 0:
            continue
        
        df = pd.concat(list_df, ignore_index=True)
        del list_df
        df = df.astype(dict_dtypes)

        df.to_parquet(path_save)
#%%
# for key in dict_dtypes.keys():
#     print(key)
#     df.astype({key:dict_dtypes[key]})
    