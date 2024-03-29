# Note: This versoin uses adjusted plant_date and harvest_date (based on photosentive)
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from icedumpy.df_tools import load_mapping, set_index_for_loc
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
    columns_s1_temporal_ideal = pd.date_range(str((datetime.datetime.strptime(df_s1_overall.columns[7], "%Y%m%d")-datetime.timedelta(days=42)).date()).replace("-", ""), df_s1_overall.columns[-1], freq="6D")
    for column in columns_s1_temporal_ideal:
        column = str(column.date()).replace("-", "")
        if not column in df_s1_overall.columns[7:]:
            df_s1_overall = df_s1_overall.assign(**{column:np.nan})
            
    df_s1_overall = df_s1_overall[df_s1_overall.columns[:7].tolist()+sorted(df_s1_overall.columns[7:])]
    return df_s1_overall
#%%
root_df_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal_version5(at-False)"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_vew_plant_info_official_polygon_disaster_all_rice_by_year_mapping(at-False)"
root_df_s1_overall = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_pixel(at-False)"
path_df_breed_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department_edited_by_ronnachai.csv"
os.makedirs(root_df_temporal, exist_ok=True)

df_breed_code = pd.read_csv(path_df_breed_code)
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
    't15': "float32",
    't16': "float32",
    't17': "float32",
    't18': "float32",
    't19': "float32",
    't20': "float32",
    't21': "float32",
    't22': "float32",
    't23': "float32",
    't24': "float32",
    't25': "float32",
    't26': "float32",
    't27': "float32",
    't28': "float32",
    't29': "float32",
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
    'photo_sensitive_f': "uint8",
    'sticky_rice_f': "uint8",
    'jasmine_rice_f': "uint8",
    'loss_ratio': 'float32',
    'polygon_area_in_square_m': 'float32',
    'row': "int32",
    'col': "int32"
}
#%%
list_strip_id = ["302", "303", "304", "305", "306", "401", "402", "403"]
for strip_id in list_strip_id:
    # Load df mapping
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
    list_p = df_mapping["p_code"].unique().tolist()
    
    for p in list_p:
        print(strip_id, p)
        path_save = os.path.join(root_df_temporal, f"df_s1ab_temporal_p{p}_s{strip_id}.parquet")
        if os.path.exists(path_save):
            continue

        # Load df vew of each p
        df_vew = pd.concat([pd.read_parquet(os.path.join(root_df_vew, file)) for file in os.listdir(root_df_vew) if file.split(".")[0].split("_")[-2][1:] in [p]], ignore_index=True)

        df_s1_overall = pd.read_parquet(os.path.join(root_df_s1_overall, f"df_s1ab_pixel_p{p}_s{strip_id}.parquet"))
        df_s1_overall = df_s1_overall.loc[df_s1_overall.ext_act_id.isin(df_mapping.ext_act_id.unique())]
        df_s1_overall = set_index_for_loc(df_s1_overall, column="ext_act_id")
        df_s1_overall.columns = [column if not "S1" in column else column[:8] for column in df_s1_overall.columns]
        df_s1_overall = add_missing_columns(df_s1_overall)
        columns_s1_temporal = np.array([datetime.datetime.strptime(column, "%Y%m%d") for column in df_s1_overall.columns[7:]])
        columns_s1_temporal = np.insert(columns_s1_temporal, 0, columns_s1_temporal[0]-datetime.timedelta(days=6))
        
        # Filter out df_vew by ext_act_id
        df_vew = df_vew[df_vew["ext_act_id"].isin(df_s1_overall["ext_act_id"])]
        
        # Merge vew with breed_code
        df_vew = pd.merge(df_vew, df_breed_code, how="inner", on="BREED_CODE")
        
        # =============================================================================
        # Change harvest date to the specific date (for photosensitive rice)
        # =============================================================================
        df_vew.loc[df_vew["photo_sensitive_f"] == 1, "harvest_date"] = df_vew.loc[df_vew["photo_sensitive_f"] == 1, "harvest_date"] + "-"+df_vew.loc[df_vew["photo_sensitive_f"] == 1, "final_plant_date"].dt.year.astype(str)
        df_vew.loc[df_vew["photo_sensitive_f"] == 1, "adjusted_harvest_date"] = pd.to_datetime(df_vew.loc[df_vew["photo_sensitive_f"] == 1, "harvest_date"])
        for row in df_vew.loc[df_vew["photo_sensitive_f"] == 1].itertuples():
            if (row.adjusted_harvest_date-row.final_plant_date).days < 0:
                df_vew.at[row.Index, "adjusted_harvest_date"]+=pd.offsets.DateOffset(years=1)
        # Also change plant date to harvest date - 180 days (for photosensitive rice)
        df_vew.loc[df_vew["photo_sensitive_f"] == 1, "adjusted_plant_date"] = df_vew.loc[df_vew["photo_sensitive_f"] == 1, "adjusted_harvest_date"] - datetime.timedelta(days=180)

        # =============================================================================
        # Change harvest date to the specific date (for non-photosensitive rice)
        # =============================================================================
        df_vew.loc[df_vew["photo_sensitive_f"] == 0, "adjusted_harvest_date"] = df_vew.loc[df_vew["photo_sensitive_f"] == 0, "final_plant_date"] + datetime.timedelta(days=180)
        df_vew.loc[df_vew["photo_sensitive_f"] == 0, "adjusted_plant_date"] = df_vew.loc[df_vew["photo_sensitive_f"] == 0, "final_plant_date"]
        
        # Filter df_vew by adjusted_plant_date (later or equal temporal first date)
        df_vew = df_vew.loc[df_vew["adjusted_plant_date"] >= datetime.datetime.strptime(df_s1_overall.columns[7], "%Y%m%d")] # This is sentinal1(A|B) first date

        # Add loss ratio
        df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
        
        # Drop deflect
        df_vew = df_vew[~df_vew["TOTAL_ACTUAL_PLANT_AREA_IN_WA"].isna()]
        
        # Want to get only the data that within the crop cycle
        list_df = []
        for vew in tqdm(df_vew.itertuples(), total=len(df_vew)):
            try:
                ext_act_id = vew.ext_act_id
                date_plant = vew.adjusted_plant_date
                date_harvest = vew.adjusted_harvest_date
                # Get column index
                column_plant = get_column_index(columns_s1_temporal, date_plant)
                column_harvest = get_column_index(columns_s1_temporal, date_harvest)

                # Get pixel data (plant -> harvest) # 30 periods
                arr_s1_overall = df_s1_overall.loc[ext_act_id]
                
                arr_s1_temporal = arr_s1_overall[df_s1_overall.columns[7:]].values
                if len(arr_s1_temporal.shape) == 1:
                    arr_s1_temporal = arr_s1_temporal.reshape(1, -1)

                arr_s1_temporal = arr_s1_temporal[:, column_plant:column_harvest]
                assert arr_s1_temporal.shape[-1] == 30

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
            except Exception as e:
                print(ext_act_id, type(e))
                pass
                
        # Skip if empty
        if len(list_df) == 0:
            continue
        
        df = pd.concat(list_df, ignore_index=True)
        del list_df
        df = df.astype(dict_dtypes)

        df.to_parquet(path_save)
#%%
