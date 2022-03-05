# Did not adjust plant_date and harvest_date
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from icedumpy.df_tools import set_index_for_loc
#%%
# Find max consecutive 1 (https://codereview.stackexchange.com/questions/138550/count-consecutive-ones-in-a-binary-list)
def len_iter(items):
    return sum(1 for _ in items)

def consecutive_one(data):
    return max(len_iter(run) for val, run in groupby(data) if val)

def add_missing_columns(df, year):
    df = df.copy()
    date_first = f"{year}0101"
    date_last = f"{year+1}1231"
    columns_ideal = pd.date_range(date_first, date_last, freq="5D")
    df = df.assign(**{str(column.date()).replace("-", ""):0 for column in columns_ideal if not str(column.date()).replace("-", "") in df.columns[7:]})
    df[list(map(lambda x: str(x.date()).replace("-", ""), columns_ideal))] = df[list(map(lambda x: str(x.date()).replace("-", ""), columns_ideal))].astype("uint8")
    df = df[df.columns[:7].tolist()+sorted(df.columns[7:])]
    return df

def get_column_index(columns, date):
    for i, column in enumerate(columns):
        if column > date:
            if i != 0:
                return i-1
            else:
                raise Exception('Out of Index')
    raise Exception('Out of Index')
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_df_gistda_flood =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_all_rice_by_year_pixel(at-False)"
root_df_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_all_rice_by_year_plot_level_features_nrt(at-False)"
os.makedirs(root_df_save, exist_ok=True)
#%%
for strip_id in np.unique([file.split(".")[0][-3:] for file in os.listdir(root_df_gistda_flood)])[1::3]:
    df_gistda_flood = pd.concat([pd.read_parquet((os.path.join(root_df_gistda_flood, file))) for file in os.listdir(root_df_gistda_flood) if file.split(".")[0][-3:] == strip_id], ignore_index=True)
    # No flood detected from GISTDA
    if len(df_gistda_flood.columns) == 7:
        continue
    df_gistda_flood.columns = df_gistda_flood.columns[:7].tolist()+df_gistda_flood.columns[7:].str.slice(0, 8).tolist()
    df_gistda_flood = set_index_for_loc(df_gistda_flood, column="ext_act_id")
    # Loop for each p_code
    for p_code in df_gistda_flood.p_code.unique():
        # Load df_vew
        df_vew = pd.concat([pd.read_parquet(os.path.join(root_df_vew, file)) for file in os.listdir(root_df_vew) if file.split(".")[0].split("_")[-2][1:] in [p_code]], ignore_index=True)
        df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
        df_vew = df_vew[~df_vew["TOTAL_ACTUAL_PLANT_AREA_IN_WA"].isna()]
        # Filter out df_vew by ext_act_id and plant_date
        df_vew = df_vew[df_vew["ext_act_id"].isin(df_gistda_flood["ext_act_id"])]
        
        for year in [2015, 2016, 2017, 2018, 2019, 2020]:
            print(strip_id, p_code, year)
            path_save = os.path.join(root_df_save, f"df_gistda_flood_plot_level_features_p{p_code}_s{strip_id}_y{year}.parquet")
            if os.path.exists(path_save):
                continue
            
            df_vew_year = df_vew[df_vew.final_plant_date.dt.year == year].copy()
            df_gistda_flood_p_code = df_gistda_flood[(df_gistda_flood["p_code"] == p_code) & (df_gistda_flood.ext_act_id.isin(df_vew_year.ext_act_id))]
            df_gistda_flood_p_code = add_missing_columns(df_gistda_flood_p_code, year=year)
        
            # Filter df_gistda flood by p_code
            columns_gistda_flood_p_code = np.array([datetime.datetime.strptime(column, "%Y%m%d") for column in df_gistda_flood_p_code.columns[7:]])
            columns_gistda_flood_p_code = np.append(columns_gistda_flood_p_code, columns_gistda_flood_p_code[-1]+datetime.timedelta(days=5))
        
            # Change harvest_date (plant_date+180 days)
            df_vew_year["final_harvest_date_stg1"] = df_vew_year["final_plant_date"]+datetime.timedelta(days=40)
            df_vew_year["final_harvest_date_stg2"] = df_vew_year["final_plant_date"]+datetime.timedelta(days=90)
            df_vew_year["final_harvest_date_stg3"] = df_vew_year["final_plant_date"]+datetime.timedelta(days=120)
            df_vew_year["final_harvest_date_stg4"] = df_vew_year["final_plant_date"]+datetime.timedelta(days=180)
            df_vew_year = set_index_for_loc(df_vew_year, column="ext_act_id")

            # Loop for each ext_act_id
            list_data = []
            for ext_act_id, df_vew_row in tqdm(df_vew_year.iterrows(), total=len(df_vew_year)):
                # Initialize dict_data (for extracted features)
                dict_data = df_vew_row.to_dict()
    
                # Loop of each growth stage
                for columns_harvest_date in ["final_harvest_date_stg1", "final_harvest_date_stg2", "final_harvest_date_stg3", "final_harvest_date_stg4"]:
                    date_plant = df_vew_row["final_plant_date"]
                    date_harvest = df_vew_row[columns_harvest_date]  
                    
                    # Get columns
                    column_plant = get_column_index(columns_gistda_flood_p_code, date_plant)
                    try:
                        column_harvest = get_column_index(columns_gistda_flood_p_code, date_harvest)+1
                    except:
                        column_harvest = len(columns_gistda_flood_p_code)
                    
                    # Get temporal (data within plant and harvest date)
                    arr_overall = df_gistda_flood_p_code.loc[ext_act_id:ext_act_id]
                    arr_temporal = arr_overall.iloc[:, column_plant+7:column_harvest+7].values
                    
                    # Find max consecutive flood (1) for each pixel
                    list_consecutive = []
                    for i in range(arr_temporal.shape[0]):
                        # Find consecutive
                        if arr_temporal[i].sum() == 0:
                            list_consecutive.append(0)
                        else:
                            list_consecutive.append(consecutive_one(arr_temporal[i]))
                        
                    # Find max consecutive flood (1) for each pixel (Relax: can be missing out for 1 period)
                    list_consecutive_relax = []
                    for i in range(arr_temporal.shape[0]):
                        # Change from [1, 0 ,1] to [1, 1, 1] (window size = 3)
                        for j in range(1, arr_temporal.shape[1]-1):
                            if (arr_temporal[i, j-1] == 1) and (arr_temporal[i, j+1] == 1):
                                arr_temporal[i, j] = 1
                        # Find consecutive
                        if arr_temporal[i].sum() == 0:
                            list_consecutive_relax.append(0)
                        else:
                            list_consecutive_relax.append(consecutive_one(arr_temporal[i]))
                                
                    # Plot-level features (normal)
                    arr_consecutive_grp = np.digitize(list_consecutive, [1, 2, 3, 4])
                    ratio_0 = (arr_consecutive_grp == 0).sum()/len(arr_consecutive_grp)
                    ratio_1 = (arr_consecutive_grp == 1).sum()/len(arr_consecutive_grp)
                    ratio_2 = (arr_consecutive_grp == 2).sum()/len(arr_consecutive_grp)
                    ratio_3 = (arr_consecutive_grp == 3).sum()/len(arr_consecutive_grp)
                    ratio_4 = (arr_consecutive_grp == 4).sum()/len(arr_consecutive_grp)
                    
                    # Plot-level features (relax)
                    arr_consecutive_relax_grp = np.digitize(list_consecutive_relax, [1, 2, 3, 4])
                    ratio_relax_0 = (arr_consecutive_relax_grp == 0).sum()/len(arr_consecutive_relax_grp)
                    ratio_relax_1 = (arr_consecutive_relax_grp == 1).sum()/len(arr_consecutive_relax_grp)
                    ratio_relax_2 = (arr_consecutive_relax_grp == 2).sum()/len(arr_consecutive_relax_grp)
                    ratio_relax_3 = (arr_consecutive_relax_grp == 3).sum()/len(arr_consecutive_relax_grp)
                    ratio_relax_4 = (arr_consecutive_relax_grp == 4).sum()/len(arr_consecutive_relax_grp)
                
                    # Make new df (Plot-level)
                    dict_data = {
                        **dict_data,
                        **{
                            "polygon_area_in_square_m": arr_overall.iat[0, 4],
                            f"flood_ratio_0_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_0,
                            f"flood_ratio_1-5_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_1,
                            f"flood_ratio_6-10_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_2,
                            f"flood_ratio_11-15_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_3,
                            f"flood_ratio_15+_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_4,
                            f"flood_ratio_relax_0_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_relax_0,
                            f"flood_ratio_relax_1-5_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_relax_1,
                            f"flood_ratio_relax_6-10_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_relax_2,
                            f"flood_ratio_relax_11-15_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_relax_3,
                            f"flood_ratio_relax_15+_{'nrt_'+columns_harvest_date.split('_')[-1]}" : ratio_relax_4}
                    }
                list_data.append(dict_data)
                
            # Skip if empty
            if len(list_data) == 0:
                continue
            # Save data
            df = pd.DataFrame(list_data)
            del list_data
            df.to_parquet(path_save)
#%%







