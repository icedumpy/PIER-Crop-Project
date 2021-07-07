import geopandas as gpd
#%%
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon2.shp")
#%%
gdf_gistda = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-gistda.shp")
gdf_pier = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon-pier.shp")
#%%
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210701\For-P-Pin\shp\thailand-tambon3.shp")
gdf.columns
