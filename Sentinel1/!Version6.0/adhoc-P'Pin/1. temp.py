import os
import numpy as np
import pandas as pd
import geopandas as gpd
#%%
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp"
root_df = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'ext_act_id', 'DANGER_TYPE', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA', 'TOTAL_DANGER_AREA_IN_WA', 'final_crop_year']
#%%
gdf_tambon = gpd.read_file(path_gdf_tambon)
#%%
for p_code in np.unique([file.split(".")[-0][-7:-5] for file in os.listdir(root_df)]):
    df = pd.concat([pd.read_parquet(os.path.join(root_df, file)) for file in os.listdir(root_df) if file.split(".")[-0][-7:-5] == p_code], ignore_index=True)
    df = df[df["final_crop_year"].isin(range(2015, 2021))]
    df.loc[df["DANGER_TYPE"] == "ฝนทิ้งช่วง", "DANGER_TYPE"] = "ภัยแล้ง"
    df.loc[df["DANGER_TYPE"].isna(), "DANGER_TYPE"] = "ไม่เกิดภัย"
    
    df["ADM3_PCODE"] = "TH"+(10000*df["PLANT_PROVINCE_CODE"]+100*df["PLANT_AMPHUR_CODE"]+df["PLANT_TAMBON_CODE"]).astype(int).astype(str)
    print(p_code, ":", df["DANGER_TYPE"].value_counts().to_dict())
    
    # Doing sth.
    for ADM3_PCODE, df_grp in df.groupby(["ADM3_PCODE"]):
        # Get area of each group
        try:
            plant_area = df_grp.groupby(["final_crop_year"]).sum()["TOTAL_ACTUAL_PLANT_AREA_IN_WA"]
            if len(df["DANGER_TYPE"].unique()) > 1:
                flood_area = df_grp[df_grp["DANGER_TYPE"] == "อุทกภัย"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
                drought_area = df_grp[df_grp["DANGER_TYPE"] == "ภัยแล้ง"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
                other_area = df_grp[~df_grp["DANGER_TYPE"].isin(["ไม่เกิดภัย", "ภัยแล้ง", "อุทกภัย"])].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
                disaster_area = df_grp[df_grp["DANGER_TYPE"] != "ไม่เกิดภัย"].groupby(["final_crop_year"]).sum()["TOTAL_DANGER_AREA_IN_WA"]
            else:
                flood_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                drought_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                other_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                disaster_area = pd.Series([], name="TOTAL_DANGER_AREA_IN_WA", dtype="float64")
                
            # Merge area of each group
            df_loss = pd.DataFrame([plant_area, flood_area, drought_area, other_area, disaster_area]).T.fillna(0)
            df_loss.columns = ["plant_area", "flood_area", "drought_area", "other_area", "disaster_area"]
            total_claims_flood = (df_loss["flood_area"] > 0).sum()
            total_claims_drought = (df_loss["drought_area"] > 0).sum()
            total_claims_other = (df_loss["other_area"] > 0).sum()
            total_claims_disaster = (df_loss["disaster_area"] > 0).sum()
            
            # Calculate the percentage (each year separately)
            df_loss = df_loss.assign(flood_area_percentage = 100*df_loss.flood_area/df_loss.plant_area)
            df_loss = df_loss.assign(drought_area_percentage = 100*df_loss.drought_area/df_loss.plant_area)
            df_loss = df_loss.assign(other_area_percentage = 100*df_loss.other_area/df_loss.plant_area)
            df_loss = df_loss.assign(disaster_area_percentage = 100*df_loss.disaster_area/df_loss.plant_area)
            
            # Convert plant area from sq.m to rai
            df_loss["plant_area"] = df_loss["plant_area"]/400
             
            # Calculate mean
            df_loss = df_loss.mean(axis=0)
            df_loss["total_claims_flood"] = total_claims_flood
            df_loss["total_claims_drought"] = total_claims_drought
            df_loss["total_claims_other"] = total_claims_other
            df_loss["total_claims_disaster"] = total_claims_disaster
            
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "plant_rai"] = df_loss["plant_area"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "flood_perc"] = df_loss["flood_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "droug_perc"] = df_loss["drought_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "other_perc"] = df_loss["other_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "disas_perc"] = df_loss["disaster_area_percentage"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "flood_totl"] = df_loss["total_claims_flood"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "droug_totl"] = df_loss["total_claims_drought"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "other_totl"] = df_loss["total_claims_other"]
            gdf_tambon.loc[gdf_tambon["ADM3_PCODE"] == ADM3_PCODE, "disas_totl"] = df_loss["total_claims_disaster"]
        except Exception as e:
            print(p_code, e)
            
gdf_tambon[['plant_rai', 'flood_perc', 'droug_perc', 'other_perc', 'disas_perc', 'flood_totl', 'droug_totl', 'other_totl', 'disas_totl']] = gdf_tambon[['plant_rai', 'flood_perc', 'droug_perc', 'other_perc', 'disas_perc', 'flood_totl', 'droug_totl', 'other_totl', 'disas_totl']].fillna(0)
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon.shp", encoding="CP874")
