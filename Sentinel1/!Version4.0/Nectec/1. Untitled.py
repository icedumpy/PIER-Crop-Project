import os
from pprint import pprint
import pandas as pd
import geopandas as gpd
#%%
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_vew_2020 = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'ext_act_id', 'DANGER_TYPE_NAME', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA', 'TOTAL_DANGER_AREA_IN_WA', 'final_crop_year']
#%%
gdf_tambon = gpd.read_file(path_gdf_tambon)
#%%
for file in os.listdir(root_df_vew):
    p_code = file.split(".")[0][-2:]
    print(p_code)
    
    # Load vew (2015-2019)
    path_file = os.path.join(root_df_vew, file)
    df = pd.read_parquet(path_file)
    df = df[columns]
    
    # Load vew (2020)
    path_file = os.path.join(root_df_vew_2020, f"vew_polygon_id_plant_date_disaster_20210202_{p_code}.parquet")
    if os.path.exists(path_file):
        df_2020 = pd.read_parquet(path_file)
        df_2020 = df_2020[columns]
    
        # Merge 2015-2019 with 2020
        df = pd.concat([df, df_2020], ignore_index=True)
        del df_2020
    
    # Keep only normal, flood and drought (2015-2020)
    df = df[df["final_crop_year"].isin(range(2015, 2021))]
    df.loc[df["DANGER_TYPE_NAME"] == "ฝนทิ้งช่วง", "DANGER_TYPE_NAME"] = "ภัยแล้ง"
    df.loc[df["DANGER_TYPE_NAME"].isna(), "DANGER_TYPE_NAME"] = "ไม่เกิดภัย"

    df["ADM3_PCODE"] = "TH"+(10000*df["PLANT_PROVINCE_CODE"]+100*df["PLANT_AMPHUR_CODE"]+df["PLANT_TAMBON_CODE"]).astype(int).astype(str)
    pprint(df["DANGER_TYPE_NAME"].value_counts().to_dict())
    print()
    
    # Doing sth.
    for ADM3_PCODE, df_grp in df.groupby(["ADM3_PCODE"]):
        # Get area of each group
        try:
            plant_area = df_grp.groupby(["final_crop_year"]).sum()["TOTAL_ACTUAL_PLANT_AREA_IN_WA"]
            if len(df["DANGER_TYPE_NAME"].unique()) > 1:
                flood_area = df_grp[df_grp["DANGER_TYPE_NAME"] == "อุทกภัย"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
                drought_area = df_grp[df_grp["DANGER_TYPE_NAME"] == "ภัยแล้ง"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
                disaster_area = df_grp[df_grp["DANGER_TYPE_NAME"] != "ไม่เกิดภัย"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
            else:
                flood_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                drought_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                disaster_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                
            # Merge area of each group
            df_loss = pd.DataFrame([plant_area, flood_area, drought_area, disaster_area]).T.fillna(0)
            df_loss.columns = ["plant_area", "flood_area", "drought_area", "disaster_area"]
            
            # Conclude into percentage (each year separately)
            df_loss = df_loss.assign(flood_area_percentage = 100*df_loss.flood_area/df_loss.plant_area)
            df_loss = df_loss.assign(drought_area_percentage = 100*df_loss.drought_area/df_loss.plant_area)
            df_loss = df_loss.assign(disaster_area_percentage = 100*df_loss.disaster_area/df_loss.plant_area)
            
            # Calculate mean
            df_loss = df_loss.mean(axis=0)
            
            # 
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "flood_perc"] = df_loss["flood_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "droug_perc"] = df_loss["drought_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "disas_perc"] = df_loss["disaster_area_percentage"]
        except Exception as e:
            print(p_code, e)
#%%
    
    