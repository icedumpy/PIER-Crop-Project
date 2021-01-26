import os
from osgeo import gdal
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
from flood_fn import load_dataframe_mapping_vew
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\vew_polygon_id_plant_date_disaster_merged"
root_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_polygon_id_rowcol_map_prov_scene_v3"
root_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV"
raster_strip_id = 402
flood_map_window_size = 2
root_save = os.path.join(r"F:\CROP-PIER\CROP-WORK\adhoc", str(raster_strip_id))
if not os.path.exists(root_save):
    os.mkdir(root_save)
#%%
print("Load df_vew")
_, df_vew =  load_dataframe_mapping_vew(root_mapping, root_vew, root_raster, raster_strip_id, flood_map_window_size)
df_vew = df_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'final_polygon', 'DANGER_TYPE_NAME']]
#%%
print("Convert dataframe to geodataframe")
gdf_vew = gpd.GeoDataFrame(df_vew)
gdf_vew['geometry'] = gdf_vew['final_polygon'].apply(wkt.loads)
gdf_vew['START_DATE'] = gdf_vew['START_DATE'].astype(str)
gdf_vew['final_plant_date'] = gdf_vew['final_plant_date'].astype(str)
gdf_vew = gdf_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'DANGER_TYPE_NAME', 'geometry']]
gdf_vew.to_file(os.path.join(root_save, "rice.shp"))
#%%
gdf_vew = gdf_vew[gdf_vew['DANGER_TYPE_NAME'] == 'อุทกภัย']
for start_date, gdf_vew_grp in tqdm(gdf_vew.groupby('START_DATE')):
    if start_date == 'NaT':
        continue
    print(os.path.join(root_save, f"flood-{start_date}.shp"))
    gdf_vew_grp.to_file(os.path.join(root_save, f"flood-{start_date}.shp"))
#%%
raster = gdal.Open(os.path.join(r"F:\CROP-PIER\CROP-WORK\Complete_VV", f"LSVVS{raster_strip_id}_2017-2019"))
for file in tqdm(os.listdir(root_save)):
    if file.endswith('.shp'):
        path_shp = os.path.join(root_save, file)
        path_rasterize = os.path.join(root_save, file.replace('.shp', '.tif'))
        command = f"gdal_rasterize -at -l {file.split('.')[0]} -burn 1.0 -ts {raster.RasterXSize:.1f} {raster.RasterYSize:.1f} -a_nodata 0.0 -te {raster.GetGeoTransform()[0]} {raster.GetGeoTransform()[3] + raster.RasterYSize*raster.GetGeoTransform()[5]} {raster.GetGeoTransform()[0] + raster.RasterXSize*raster.GetGeoTransform()[1]} {raster.GetGeoTransform()[3]} -ot Byte -of GTiff" + " {} {}".format(path_shp.replace('\\', '/'), path_rasterize.replace('\\', '/'))
        os.system(command)
    










