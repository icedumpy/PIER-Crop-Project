import os
import numpy as np
import pandas as pd
import geopandas as gpd
from icedumpy.geo_tools import convert_to_geodataframe
#%%
path_province_code = r"F:\CROP-PIER\Province_code_csv.csv"
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp"
root_df = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
columns = ['PLANT_PROVINCE_CODE', 'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 
           'ext_act_id', 'DANGER_TYPE', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA', 'TOTAL_DANGER_AREA_IN_WA', 'final_crop_year', 
           'tambon_key', 'geometry']
#%%
df_province_code = pd.read_csv(path_province_code)
df_province_code = df_province_code[df_province_code["full_tambon_name"].isin([
    # "สกลนคร_วานรนิวาส_ขัวก่าย", 
    # "ร้อยเอ็ด_เมืองสรวง_คูเมือง",
    # "ร้อยเอ็ด_ปทุมรัตต์_ดอกล้ำ",
    # "ขอนแก่น_แวงใหญ่_คอนฉิม",
    "นครราชสีมา_ขามสะแกแสง_ขามสะแกแสง",
    # "สระแก้ว_ตาพระยา_ทัพราช",
    "นครสวรรค์_ท่าตะโก_สายลำโพง",
    # "อุทัยธานี_สว่างอารมณ์_ไผ่เขียว",
    # "ชัยนาท_เนินขาม_เนินขาม"
])]

gdf_tambon = gpd.read_file(path_gdf_tambon)
gdf_tambon["tambon_key"] = gdf_tambon["ADM3_PCODE"].str.slice(2, 4).astype(int).astype(str) + "_" + gdf_tambon["ADM3_PCODE"].str.slice(4, 6).astype(int).astype(str) + "_" + gdf_tambon["ADM3_PCODE"].str.slice(6, 8).astype(int).astype(str)
gdf_tambon = gdf_tambon[gdf_tambon["tambon_key"].isin(df_province_code["tambon_key"])]
#%%
for tambon_key in df_province_code["tambon_key"].unique():
    print(tambon_key)
    
    # Load and filter df
    p_code = tambon_key[:2]
    df = pd.concat([pd.read_parquet(os.path.join(root_df, file)) for file in os.listdir(root_df) if file.split(".")[-0][-7:-5] == p_code], ignore_index=True)
    df = df[df["final_crop_year"].isin(range(2015, 2021))]
    df.loc[df["DANGER_TYPE"] == "ฝนทิ้งช่วง", "DANGER_TYPE"] = "ภัยแล้ง"
    df.loc[df["DANGER_TYPE"].isna(), "DANGER_TYPE"] = "ไม่เกิดภัย"
    df["tambon_key"] = df["PLANT_PROVINCE_CODE"].astype(str) + "_" + df["PLANT_AMPHUR_CODE"].astype(str) + "_"+ df["PLANT_TAMBON_CODE"].astype(str)
    df = df[df["tambon_key"] == tambon_key]
    
    # Convert to GeoDataFrame and filter within tambon
    gdf = convert_to_geodataframe(df, polygon_column="polygon")
    gdf = gdf[gdf.centroid.within(gdf_tambon[gdf_tambon["tambon_key"] == tambon_key].geometry.iloc[0])]
    df = df.loc[gdf.index]
#%%