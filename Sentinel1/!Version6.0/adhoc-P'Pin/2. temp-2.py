import pandas as pd
import geopandas as gpd
#%%
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon.shp"
path_gdf_s1ab = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\sen1ab.shp"
path_gdf_hl8s2 = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\harmonized-ls8-s2.shp"
#%%
gdf_tambon = gpd.read_file(path_gdf_tambon, encoding="CP874")
gdf_s1ab = gpd.read_file(path_gdf_s1ab)
gdf_s1ab = gdf_s1ab.to_crs(4326)
gdf_hl8s2 = gpd.read_file(path_gdf_hl8s2)
#%% Modify something
# Drop some columns of tambon
gdf_tambon = gdf_tambon[["ADM1_TH", "ADM2_TH", "ADM3_TH", 'plant_rai', 'flood_perc', 'droug_perc', 'other_perc', 'disas_perc', 'flood_totl', 'droug_totl', 'other_totl', 'disas_totl', 'geometry']]

# Union s1ab, hl8s2
gdf_s1ab = gpd.GeoDataFrame([gdf_s1ab.unary_union])
gdf_s1ab.columns = ["geometry"]

gdf_hl8s2 = gpd.GeoDataFrame(gdf_hl8s2.unary_union)
gdf_hl8s2.columns = ["geometry"]
#%%
gdf_tambon["isin_hl8s2"] = gdf_hl8s2.geometry.apply(lambda val: gdf_tambon.geometry.within(val)).T.any(axis=1)
gdf_tambon["isin_s1ab"] = gdf_s1ab.geometry.apply(lambda val: gdf_tambon.geometry.within(val)).T.any(axis=1)
#%%
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon2.shp", encoding="CP874")
#%%
df_tambon = gdf_tambon[["ADM1_TH", "ADM2_TH", "ADM3_TH", 'plant_rai', 'flood_perc', 'droug_perc', 'other_perc', 'disas_perc', 'flood_totl', 'droug_totl', 'other_totl', 'disas_totl', "isin_hl8s2", "isin_s1ab"]]
df_tambon["disas_totl"] = df_tambon["disas_totl"].astype("uint8")
df_tambon.to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\stats.csv", encoding="CP874")