import os
import pandas as pd
import geopandas as gpd
#%%
path_p_code = r"F:\CROP-PIER\Province_code_csv.csv"
path_rice_code = r"F:\CROP-PIER\CROP-WORK\rice_age_from_rice_department.csv"
path_shp1 = r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\shp-123\เพชรบูรณ์.shp"
path_shp2 = r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\shp-123\โคราช.shp"
path_shp3 = r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\shp-123\ขอนแก่น.shp"
path_shp4 = r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\shp-123\ทุ่งกุลา.shp"
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
#%%
df_p_code = pd.read_csv(path_p_code)
df_rice_code = pd.read_csv(path_rice_code, encoding="TIS-620")
df_selected_area = pd.concat([gpd.read_file(path) for path in [path_shp1, path_shp2, path_shp3, path_shp4]], ignore_index=True)

df_p_code = df_p_code.rename(columns={
    "PROVINCE_CODE": "PLANT_PROVINCE_CODE", 
    "AMPHUR_CODE":"PLANT_AMPHUR_CODE",
    "TAMBON_CODE":"PLANT_TAMBON_CODE"
})
df_p_code = df_p_code[['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE','PLANT_TAMBON_CODE', 'PROVINCE_NAME', 'AMPHUR_NAME', 'TAMBON_NAME']]
#%%
list_p = df_selected_area["ADM1_PCODE"].str.slice(2,).unique().tolist()
#%%
df_vew = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if file.split(".")[0][-7:-5] in list_p], ignore_index=True)
#%% Load vew and filter by selected_tambon
df_vew = df_vew.assign(ADM3_PCODE = "TH"+df_vew["PLANT_PROVINCE_CODE"].astype(str).str.zfill(2)+df_vew["PLANT_AMPHUR_CODE"].astype(str).str.zfill(2)+df_vew["PLANT_TAMBON_CODE"].astype(str).str.zfill(2))
df_vew = df_vew.loc[df_vew["ADM3_PCODE"].isin(df_selected_area["ADM3_PCODE"].unique())]
#%%

#%%
# Add breed name
df_vew = pd.merge(df_vew, df_rice_code[["BREED_CODE", "BREED_NAME"]], on="BREED_CODE", how="inner")
# Add p, a, t name
df_vew = pd.merge(df_vew, df_p_code, on=['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE','PLANT_TAMBON_CODE'], how="inner")

# Add final_plant_year month date
df_vew["final_plant_date"] = pd.to_datetime(df_vew["final_plant_date"])
df_vew = df_vew.assign(final_plant_year=df_vew["final_plant_date"].dt.year)
df_vew = df_vew.assign(final_plant_month=df_vew["final_plant_date"].dt.month)

# Add in_season
df_vew = df_vew.assign(in_season_rice_f=df_vew['final_plant_date'].dt.month.between(5, 10).astype("int"))

# Add danger group
df_vew.loc[df_vew["DANGER_TYPE_NAME"].isin(["อุทกภัย"]), "DANGER_TYPE_GRP"] = "Flood"
df_vew.loc[df_vew["DANGER_TYPE_NAME"].isin(["ภัยแล้ง", "ฝนทิ้งช่วง"]), "DANGER_TYPE_GRP"] = "Drought"
df_vew.loc[(~df_vew["DANGER_TYPE_NAME"].isin(["อุทกภัย", "ภัยแล้ง", "ฝนทิ้งช่วง"])) & (~df_vew["DANGER_TYPE_NAME"].isna()), "DANGER_TYPE_GRP"] = "Other"

#%%
dict_columns_new_name = {
    "PROFILE_CENTER_ID":"ext_profile_center_id",
    "PLANT_PROVINCE_CODE":"plant_province_code",
    "PLANT_AMPHUR_CODE":"plant_amphur_code",
    "PLANT_TAMBON_CODE":"plant_tambon_code",
    "PROVINCE_NAME":"plant_province_name",
    "AMPHUR_NAME":"plant_amphur_name",
    "TAMBON_NAME":"plant_tambon_name",
    "ext_act_id":"ext_act_id",
    "TYPE_CODE":"type_code",
    "BREED_CODE":"breed_code",
    "BREED_NAME":"breed_name",
    "ACT_RAI_ORI":"plant_are_rai",
    "ACT_NGAN_ORI":"plant_are_ngan",
    "ACT_WA_ORI":"plant_are_wa",
    "TOTAL_ACTUAL_PLANT_AREA_IN_WA":"total_plant_area_in_wa",
    "final_plant_year":"final_plant_year",
    "final_plant_month":"final_plant_month",
    "final_plant_date":"final_plant_date",
    "in_season_rice_f":"in_season_rice_f",
    "final_polygon":"polygon",
    "START_DATE":"danger_date",
    "DANGER_TYPE_NAME":"danger_type",
    "DANGER_TYPE_GRP":"danger_type_grp",
    "DANGER_TOTAL_RAI":"danger_total_rai",
    "DANGER_TOTAL_NGAN":"danger_total_ngan",
    "DANGER_TOTAL_WA":"danger_total_wa",
    "TOTAL_DANGER_AREA_IN_WA":"total_danger_area_in_wa"
}
df_vew = df_vew.rename(columns=dict_columns_new_name)
df_vew = df_vew[list(dict_columns_new_name.values())]

# Additional drop: case total_plant_area_in_wa == nan
df_vew = df_vew[~df_vew["total_plant_area_in_wa"].isna()]
# Additional drop: case danger but danger_area == 0
df_vew = df_vew.loc[~((df_vew["total_danger_area_in_wa"] == 0) & (~df_vew["danger_type"].isna()))]

# Change dtypes
df_vew = df_vew.astype({
    "ext_profile_center_id":"int64",
    "ext_act_id":"int64",
    "breed_code":"int64",
    "plant_are_rai":"int64",
    "plant_are_ngan":"int64",
    "plant_are_wa":"int64",
    "total_plant_area_in_wa":"int64",
})
#%%
df_vew.head(50000).to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\Data2\Data-for-NECTEC-20210528.csv", index=False)
df_vew.head(50000).to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\Data2\Data-for-NECTEC-20210528.txt", index=False, sep='\t', mode='w')
#%%

#%%
df_vew["plant_province_name"].value_counts().sort_values()


    





















