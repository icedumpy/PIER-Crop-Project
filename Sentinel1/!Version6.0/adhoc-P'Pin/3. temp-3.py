import pandas as pd
import geopandas as gpd
#%%
path_df = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\Tambon_stats_exgratia.csv"
path_gdf_tambon = r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon.shp"
#%%
df = pd.read_csv(path_df)
gdf_tambon = gpd.read_file(path_gdf_tambon, encoding="CP874")
#%%
df = df.rename(columns={"province":"ADM1_TH", "amphur":"ADM2_TH", "tambon":"ADM3_TH"})
df = df[['ADM1_TH', 'ADM2_TH', 'ADM3_TH', 'totalyrs', 'meanarea']]
#%%
gdf_tambon = pd.merge(gdf_tambon, df, how="left", on=["ADM1_TH", "ADM2_TH", "ADM3_TH"])
gdf_tambon[["totalyrs", "meanarea"]] = gdf_tambon[["totalyrs", "meanarea"]].fillna(0)
gdf_tambon.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon3.shp", encoding="CP874")
#%%

