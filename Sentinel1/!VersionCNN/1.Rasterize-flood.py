import os
import datetime
import rasterio
from icedumpy.geo_tools import convert_to_geodataframe, gdal_rasterize
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping
#%%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    # Define paths
    print(strip_id)
    path_raster = rf"C:\Users\PongporC\Desktop\temp\S1AB\S1AB_{strip_id}.vrt"
    root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
    root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
    root_rasterized = r"G:\!PIER\!Flood-map-groundtruth\Sentinel-1\S1AB"
    
    # Load mapping (to find which polygon belongs to the selected strip_id)
    df_mapping, list_p = load_mapping(root_df_mapping, strip_id=strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & (df_mapping["is_within"])]
    
    # Load raster
    raster = rasterio.open(path_raster)
    raster_reso = raster.get_transform()[1]
    raster_bounds = list(raster.bounds)
    descriptions = [datetime.datetime.strptime(item[:8], "%Y%m%d") for item in raster.descriptions if item not in ["row", "col"]]
    
    # Load vew, clean, keep only flood
    df_vew = clean_and_process_vew(load_vew(root_df_vew, list_p=list_p), list_new_polygon_id=df_mapping["new_polygon_id"])
    df_vew = df_vew.loc[(df_vew["START_DATE"] >= descriptions[0])]
    df_vew = df_vew.loc[df_vew["DANGER_TYPE_NAME"] == "อุทกภัย"]

    # Loop for each raster date
    for i in range(len(descriptions)-1):
        
        # Filter polygons (within flood date (date0 < x <= date1))
        df_vew_selected = df_vew.loc[(df_vew["START_DATE"] > descriptions[i]) & (df_vew["START_DATE"] <= descriptions[i+1])]
        if len(df_vew_selected) == 0:
            continue
        else:
            # Save shp file
            path_tmp_shp = r"C:\Users\PongporC\Desktop\temp\temp_for_rasterize.shp"
            gdf = convert_to_geodataframe(df_vew_selected)
            gdf = gdf[["loss_ratio", "geometry"]]
            gdf.to_file(path_tmp_shp)
        
            # Rasterize
            path_rasterized = os.path.join(root_rasterized, strip_id, f"{str(descriptions[i+1].date())}.tif")
            if not os.path.exists(os.path.dirname(path_rasterized)):
                os.makedirs(os.path.dirname(path_rasterized), exist_ok=True)
            gdal_rasterize(path_tmp_shp, path_rasterized,
                       xRes=raster_reso,
                       yRes=raster_reso,
                       outputBounds=raster_bounds,
                       format="GTiff",
                       outputType="Float32",
                       burnField="loss_ratio",
                       allTouched=False,
                       noData=0)
            
        
        
        
        
        
        
        
        
        
        
        
        