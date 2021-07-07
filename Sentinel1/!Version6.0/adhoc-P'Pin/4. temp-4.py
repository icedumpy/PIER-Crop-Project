import pandas as pd
import geopandas as gpd
#%%
path_df = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\Tambon_stats_final.csv"
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon.shp"
#%%
df = pd.read_csv(path_df)
gdf_tambon = gpd.read_file(path_gdf_tambon, encoding="CP874")
df = df[df["PIER_Prefer"] == 1]
df = df.rename(columns={"province":"ADM1_TH", "amphur":"ADM2_TH", "tambon":"ADM3_TH"})
df = df[['ADM1_TH', 'ADM2_TH', 'ADM3_TH', 'PIER_Prefer', 'GISTDA', 'lossYrs_allrisk']]
gdf_tambon = pd.merge(gdf_tambon, df, how="left", on=["ADM1_TH", "ADM2_TH", "ADM3_TH"])
gdf_tambon[["lossYrs_allrisk"]] = gdf_tambon[["lossYrs_allrisk"]].fillna(0)
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-pier.shp", encoding="CP874")
#%%
df = pd.read_csv(path_df)
gdf_tambon = gpd.read_file(path_gdf_tambon, encoding="CP874")
df = df[df["GISTDA"] == 1]
df = df.rename(columns={"province":"ADM1_TH", "amphur":"ADM2_TH", "tambon":"ADM3_TH"})
df = df[['ADM1_TH', 'ADM2_TH', 'ADM3_TH', 'PIER_Prefer', 'GISTDA', 'lossYrs_allrisk']]
gdf_tambon = pd.merge(gdf_tambon, df, how="left", on=["ADM1_TH", "ADM2_TH", "ADM3_TH"])
gdf_tambon[["lossYrs_allrisk"]] = gdf_tambon[["lossYrs_allrisk"]].fillna(0)
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-gistda.shp", encoding="CP874")
#%%
# =============================================================================
# Ver 2 - final
# =============================================================================
path_df = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\Tambon_stats_final.csv"
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon3.shp"
#%%
df = pd.read_csv(path_df)
gdf_tambon = gpd.read_file(path_gdf_tambon, encoding="CP874")
gdf_tambon_s1b_hl8s2 = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon2.shp", encoding="CP874")
df = df[(df["PIER_Prefer"] == 1) | (df["GISTDA"] == 1)]
df = df.rename(columns={"province":"ADM1_TH", "amphur":"ADM2_TH", "tambon":"ADM3_TH"})
df = df[['ADM1_TH', 'ADM2_TH', 'ADM3_TH', 'PIER_Prefer', 'GISTDA', 'lossYrs_allrisk']]
gdf_tambon = pd.merge(gdf_tambon, df, how="left", on=["ADM1_TH", "ADM2_TH", "ADM3_TH"])
gdf_tambon[["lossYrs_allrisk"]] = gdf_tambon[["lossYrs_allrisk"]].fillna(0)
gdf_tambon[["GISTDA"]] = gdf_tambon[["GISTDA"]].fillna(0)
gdf_tambon[["PIER_Prefer"]] = gdf_tambon[["PIER_Prefer"]].fillna(0)
gdf_tambon = pd.concat([gdf_tambon, gdf_tambon_s1b_hl8s2[["isin_hl8s2", "isin_s1ab"]]], axis=1)
#%%
gdf_tambon = gdf_tambon[[
    "ADM1_TH", "ADM2_TH", "ADM3_TH", 'plant_rai', 'flood_perc', 'droug_perc', 'other_perc', 'disas_perc',
    'flood_totl', 'droug_totl', 'other_totl', 'disas_totl', 'totalyrs',
    'meanarea', 'PIER_Prefer', 'GISTDA', 'isin_hl8s2', 'isin_s1ab', 'geometry'
]]
#%%
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-final.shp", encoding="CP874")
#%%
gdf_tambon = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-final.shp")
gdf_tambon.columns
#%%