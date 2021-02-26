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

def get_column_index(columns_s1_temporal, date):
    for i, column_s1_temporal in enumerate(columns_s1_temporal):
        if column_s1_temporal >= date:
            if i != 0:
                return i-1
            else:
                raise Exception('Out of Index')
    raise Exception('Out of Index')

def add_missing_columns(df_s1_overall):
    columns_s1_temporal_ideal = pd.date_range(df_s1_overall.columns[7], df_s1_overall.columns[-1], freq="6D")
    for column in columns_s1_temporal_ideal:
        column = str(column.date()).replace("-", "")
        if not column in df_s1_overall.columns[7:]:
            df_s1_overall = df_s1_overall.assign(**{column:np.nan})
    df_s1_overall.iloc[:, 7:] = df_s1_overall[df_s1_overall.columns[7:]].reindex(sorted(df_s1_overall.columns[7:]), axis=1)
    return df_s1_overall
#%%
root_df_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_overall = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
os.makedirs(root_df_temporal, exist_ok=True)
#%%
list_strip_id = ["302", "303", "304", "305", "306", "401", "402", "403"]
for strip_id in list_strip_id[4::5]:
    # Load df mapping
    df_mapping, _ = load_mapping(root_df_mapping, strip_id = strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
    list_p = df_mapping["p_code"].unique().tolist()
    
    # Set dtypes
    dict_dtypes = {f"t{i}":"float32" for i in range(31)}
    dict_dtypes["new_polygon_id"] = "int32"
    dict_dtypes["PLANT_PROVINCE_CODE"] = "int32"
    dict_dtypes["PLANT_AMPHUR_CODE"] = "int32"
    dict_dtypes["PLANT_TAMBON_CODE"] = "int32"
    dict_dtypes["ext_act_id"] = "int64"
    dict_dtypes["BREED_CODE"] = "float32"
    dict_dtypes["loss_ratio"] = "float32"
    dict_dtypes["polygon_area_in_square_m"] = "float32"
    dict_dtypes["row"] = "int32"
    dict_dtypes["col"] = "int32"

    for p in list_p:
        print(strip_id, p)
        path_save = os.path.join(root_df_temporal, f"df_s1ab_temporal_p{p}_s{strip_id}.parquet")
        if os.path.exists(path_save):
            continue

        # Load df vew of each p
        df_vew = load_vew(root_df_vew, list_p=[p])
        df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())

        df_s1_overall = pd.read_parquet(os.path.join(root_df_s1_overall, f"df_s1ab_pixel_p{p}_s{strip_id}.parquet"))
        df_s1_overall = df_s1_overall.loc[df_s1_overall.new_polygon_id.isin(df_mapping.new_polygon_id.unique())]
        df_s1_overall = set_index_for_loc(df_s1_overall, column="new_polygon_id")
        df_s1_overall.columns = [column if not "S1" in column else column[:8] for column in df_s1_overall.columns]
        df_s1_overall = add_missing_columns(df_s1_overall)
        columns_s1_temporal = np.array([datetime.datetime.strptime(column, "%Y%m%d") for column in df_s1_overall.columns[7:]])
        columns_s1_temporal = np.insert(columns_s1_temporal, 0, columns_s1_temporal[0]-datetime.timedelta(days=6))
        
        # Filter df_vew by final_plant_date (later or equal temporal first date)
        df_vew = df_vew.loc[df_vew["final_plant_date"] >= datetime.datetime.strptime(df_s1_overall.columns[7], "%Y%m%d")] # This is sentinal1(A|B) first date

        # Change harvest date to plant date + 180 days
        df_vew["final_harvest_date"] = df_vew["final_plant_date"] + datetime.timedelta(days=180)

        # Want to get only the data that within the crop cycle (plantdate to plantdate+180days)
        list_df = []
        for vew in tqdm(df_vew.itertuples(), total=len(df_vew)):
            try:
                new_polygon_id = vew.new_polygon_id
                date_plant = vew.final_plant_date
                date_harvest = vew.final_harvest_date
                # Get column index
                column_plant = get_column_index(columns_s1_temporal, date_plant)
                column_harvest = get_column_index(columns_s1_temporal, date_harvest)

                # Get pixel data (plant -> harvest) # 31 periods
                arr_s1_overall = df_s1_overall.loc[new_polygon_id]

                arr_s1_temporal = arr_s1_overall[df_s1_overall.columns[7:]].values
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
                pass

        df = pd.concat(list_df, ignore_index=True)
        del list_df
        df = df.astype(dict_dtypes)

        df.to_parquet(path_save)
#%%
