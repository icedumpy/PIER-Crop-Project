#%% Import libraries
import os
import random
import pyproj
from shapely.ops import transform
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm
from rasterio.mask import mask
from icedumpy.df_tools import load_vew
from icedumpy.geo_tools import create_polygon_from_wkt, create_vrt, create_tiff
#%% Import functions
project = pyproj.Transformer.from_crs(
    'epsg:4326',     # source coordinate system ex. 'epsg:4326'
    'epsg:32647',  # destination coordinate system ex. 'epsg:32647'
    always_xy=True # Must have
).transform

def create_valid_mask(root_raster_valid_mask, root_raster, strip_id):
    band_count = 0
    for root, _, files in os.walk(root_raster):
        if os.path.basename(root) == strip_id:
            for file in files:
                if file.endswith(".tif") and (random.random() >= 0.75):
                    raster = rasterio.open(os.path.join(root, file))
                    raster_im = raster.read(1)
                    if "valid_mask" not in locals():
                        valid_mask = np.zeros((raster.height, raster.width), 'uint8')
                    valid_mask+=(raster_im > 0)
                    band_count+=1
                    
    valid_mask = (valid_mask > (0.7*band_count)).astype('uint8')
    
    create_tiff(path_save=os.path.join(root_raster_valid_mask, f"s1_valid_mask_{strip_id}.tif"), 
                im=np.expand_dims(valid_mask, axis=0),
                projection=raster.crs.to_wkt(), 
                geotransform=raster.get_transform(), 
                drivername="GTiff",
                dtype="uint8")

def get_pixel_location_from_coords(x, y, geotransform):
    # Verify check
    ''' x : lon/easting
        y : lat/northing
        geotransform : geotransform of raster 
    '''
    divided_by = (geotransform[1]*geotransform[5])-(geotransform[2]-geotransform[4])
    pixel_x = ((geotransform[5]*(x-geotransform[0]))-(geotransform[2]*(x-geotransform[3])))/divided_by
    pixel_y = ((geotransform[1]*(y-geotransform[3]))-(geotransform[4]*(y-geotransform[0])))/divided_by
    pixel_x = int(np.floor(pixel_x))
    pixel_y = int(np.floor(pixel_y))
    return pixel_x, pixel_y # col, row
#%%
root_raster = r"G:\!PIER\!FROM_2TB\Complete_VV_separate"
root_raster_valid_mask = r"G:\!PIER\!FROM_2TB\s1_valid_mask"
root_raster_rowcol = r"G:\!PIER\!FROM_2TB\s1_pixel_rowcol_map"
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_ext_act_id_rowcol_map_prov_scene_v5(at-False)_2020"
path_gdf_provinces = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-province.shp"

# For getting list_p from strip_id
root_df_mapping_atTrue = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v3(at-True)"

os.makedirs(root_df_mapping, exist_ok=True)
gdf_provinces = gpd.read_file(path_gdf_provinces)
#%%
list_strip_id = ["101", "102", "103", "104", "105", "106", "107", "108", "109", 
                 "201", "202", "203", "204", "205", "206", "207", "208",
                 "301", "302", "303", "304", "305", "306",
                 "401", "402", "403"]
pbar_1 = tqdm(list_strip_id)

for strip_id in pbar_1:
    pbar_1.set_description(f"s{strip_id}")
    # Load raster_rowcol, raster_valid_mask
    # Create virtual rowcol raster
    
    tqdm.write("\nCreating raster_rowcol")
    path_raster_rowcol = os.path.join(root_raster_rowcol, f"s1_pixel_rowcol_map_{strip_id}.vrt")
    create_vrt(path_save=path_raster_rowcol, 
               list_path_raster=[os.path.join(root_raster_rowcol, f"s1_pixel_row_map_{strip_id}.tiff"), 
                                 os.path.join(root_raster_rowcol, f"s1_pixel_col_map_{strip_id}.tiff")])
    tqdm.write("\nLoading raster_rowcol")
    raster_rowcol = rasterio.open(path_raster_rowcol)
    geotransform = raster_rowcol.get_transform()
    
    # Create valid mask raster (if not available)
    path_raster_valid_mask = os.path.join(root_raster_valid_mask, f"s1_valid_mask_{strip_id}.tif")
    if not os.path.exists(path_raster_valid_mask):
        tqdm.write("\nCreating raster_valid_mask")
        create_valid_mask(root_raster_valid_mask, root_raster, strip_id)
    tqdm.write("\nLoading raster_valid_mask")
    raster_valid_mask = rasterio.open(path_raster_valid_mask)
    raster_valid_mask_im = raster_valid_mask.read(1)
    
    # Get list_p of the selected strip_id
    list_p = [file.split(".")[0].split("_")[-2][1:] for file in os.listdir(root_df_mapping_atTrue) if file.split(".")[0].split("_")[-1][1:] == strip_id]
    pbar_2 = tqdm(list_p)
    
    # Load df_vew and drop duplicate ext_act_id
    for p in pbar_2:
        pbar_2.set_description(f"s{strip_id}_p{p}")
        path_df_mapping = os.path.join(root_df_mapping, f"df_s1_ext_act_id_rowcol_map_p{p}_s{strip_id}.parquet")
        if os.path.exists(path_df_mapping):
            tqdm.write(f"df_mapping of strip_id_p: s{strip_id}_p{p} is already exists")
            continue
        polygon_province = gdf_provinces.loc[gdf_provinces["ADM1_PCODE"] == f"TH{p}"].geometry.values[0]
        polygon_province.simplify(0.01)
        
        df_vew = load_vew(root_df_vew, p)
        
        tqdm.write("\nStart creating df_mapping")
        list_df_mapping_T1 = []
        list_df_mapping_T2 = []
        for df_vew_row in df_vew.itertuples():
            polygon = create_polygon_from_wkt(df_vew_row.final_polygon)
            try:
                # Get mapping row, col
                masked, _ = mask(raster_rowcol, polygon, crop=True, nodata=-99, all_touched=False)
                
                # If can't mask(polygon is too small), get only one pixel at the centroid of the polygon
                if (masked == -99).all():
                    cols, rows = [[item] for item in get_pixel_location_from_coords(polygon.centroid.x, polygon.centroid.y, geotransform)]
                    # Skip if row or col is negative
                    if cols[0] < 0 or rows[0] < 0:
                        continue
                else:
                    rows = masked[0][masked[0] != -99].astype("uint16")
                    cols = masked[1][masked[1] != -99].astype("uint16")
                
                # Check if rows, cols are valid or not
                valid_pixels = raster_valid_mask_im[rows, cols]
                
                # Skip if all masked_pixels are not valid
                if valid_pixels.sum() == 0:
                    continue
                
                # Calculate diff_ratio (polygon_area/reported_area)
                polygon_area = transform(project, polygon).area
                diff_ratio = polygon_area/(4*df_vew_row.TOTAL_ACTUAL_PLANT_AREA_IN_WA)
                
                # Check is within
                if not polygon.is_valid:
                    continue
                
                is_within = polygon.within(polygon_province)
                
                # Tier1: polygon
                if 0.75 <= diff_ratio <= 1.25:
                    list_df_mapping_T1.append(pd.DataFrame({"ext_act_id":df_vew_row.ext_act_id,
                                                            "p_code":p,
                                                            "row":rows, 
                                                            "col":cols,
                                                            "polygon_area_in_square_m":polygon_area,
                                                            "is_within":is_within}))
                # Tier2: polygon
                else:
                    list_df_mapping_T2.append(pd.DataFrame({"ext_act_id":df_vew_row.ext_act_id,
                                                            "p_code":p,
                                                            "row":rows, 
                                                            "col":cols,
                                                            "polygon_area_in_square_m":polygon_area,
                                                            "is_within":is_within}))
                    
            except (ValueError, IndexError, ZeroDivisionError):
                # (Polygon of raster bounds, Valid_pixels out of bounds, Reported area == 0)
                continue
        
        try:
            df_mapping_T1 = pd.concat(list_df_mapping_T1, ignore_index=True)
            df_mapping_T1 = df_mapping_T1.assign(tier = 1)
        except:
            df_mapping_T1 = pd.DataFrame()
        
        try:
            df_mapping_T2 = pd.concat(list_df_mapping_T2, ignore_index=True)
            df_mapping_T2 = df_mapping_T2.assign(tier = 2)
        except:
            df_mapping_T2 = pd.DataFrame()
        
        df_mapping = pd.concat([df_mapping_T1, df_mapping_T2], ignore_index=True)
        
        if len(df_mapping) != 0:
            tqdm.write("\nSaving df_mapping")
            df_mapping.to_parquet(path_df_mapping)
























