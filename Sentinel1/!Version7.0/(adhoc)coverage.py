import os
import pandas as pd
import geopandas as gpd
#%%
path_s1 = r"F:\CROP-PIER\COPY-FROM-PIER\Sentinel1A_Index\Sentinel1_Index.shp"
path_hls = r"F:\CROP-PIER\CROP-WORK\hls_s2_tiles\hls_s2_tiles.shp"
path_thailand = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-province.shp"
#%%
gdf_s1 = gpd.read_file(path_s1)
gdf_hls = gpd.read_file(path_hls)
gdf_thailand = gpd.read_file(path_thailand, encoding="cp874")
#%%
gdf_s1 = gdf_s1[gdf_s1["SID"].isin(["S302", "S303", "S304", "S305", 
                                    "S306", "S401", "S402", "S403"])]
gdf_s1 = gdf_s1.to_crs({"init":"EPSG:4326"})
#%%
gdf_thailand_s1 = gpd.overlay(gdf_thailand, gdf_s1, how='intersection')
gdf_thailand_hls = gpd.overlay(gdf_thailand, gdf_hls, how='intersection')
#%%
gdf_thailand.loc[gdf_thailand["ADM1_EN"].isin(gdf_thailand_s1["ADM1_EN"].unique()), "S1"] = True
gdf_thailand.loc[gdf_thailand["ADM1_EN"].isin(gdf_thailand_hls["ADM1_EN"].unique()), "HLS"] = True
#%%
gdf_thailand.iloc[:, -2:] = gdf_thailand.iloc[:, -2:].fillna(value=False)
#%%
df_thailand = gdf_thailand[["ADM1_EN", "ADM1_TH", "ADM1_PCODE", "S1", "HLS"]]
#%%
df_thailand.to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20210921\provice-s1-hls.csv", encoding="cp874", index=False)
#%%
ice = pd.read_csv(r"F:\CROP-PIER\Province_code_csv.csv")
ice = ice.drop_duplicates("PROVINCE_CODE")
ice = ice.sort_values("PROVINCE_CODE")
