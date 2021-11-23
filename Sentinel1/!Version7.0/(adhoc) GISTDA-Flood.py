import os
import re
import datetime
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from icedumpy.df_tools import set_index_for_loc
from icedumpy.geo_tools import gdal_rasterize, create_vrt, create_tiff
#%%
root = r"F:\CROP-PIER\CROP-WORK\GISTDA-Flood"
root_temp = r"C:\Users\PongporC\Desktop\temp\Flood-map"
root_final = r"F:\CROP-PIER\CROP-WORK\GISTDA-Flood\Rasterized"
root_raster_temp = r"G:\!PIER\!FROM_2TB\s1_valid_mask"
path_sen1a_template = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\sen1a_strip_id_template.parquet"
path_sen1a_index = r"F:\CROP-PIER\COPY-FROM-PIER\Sentinel1A_Index\Sentinel1_Index_4326.shp"

df_sen1a_template = pd.read_parquet(path_sen1a_template)
df_sen1a_template = set_index_for_loc(df_sen1a_template, column="strip_id")
df_sen1a_index = gpd.read_file(path_sen1a_index)
#%% List all GISTDA files
list_file = []
for r, _, files in os.walk(root):
    for file in files:
        if file.endswith(".shp"):
            date = datetime.datetime.strptime(re.search(r'\d{8}', file).group(), '%Y%m%d').date()
            year = date.year
            month = date.month
            path = os.path.join(r, file)
            list_file.append({
                "filename" : file,
                "path" : path,
                "date" : date,
                "year" : year,
                "month": month
            })
#%% Group by range of date (freq = 5 days)
df_file = pd.DataFrame(list_file)
df_date_range = pd.DataFrame(
    pd.date_range(
        start=df_file["date"].min(),
        end=df_file["date"].max(), 
        freq="5D"
    ), columns=["start"]
)
df_date_range["stop"] = df_date_range["start"]+datetime.timedelta(days=4)

group_number = 1
for row in df_date_range.itertuples(index=False):
    if ((df_file["date"] >= row[0]) & (df_file["date"] <= row[1])).any():
        df_file.loc[(df_file["date"] >= row[0]) & (df_file["date"] <= row[1]), "group_number"] = group_number
        df_file.loc[(df_file["date"] >= row[0]) & (df_file["date"] <= row[1]), "first_date"] = row[0]
        df_file.loc[(df_file["date"] >= row[0]) & (df_file["date"] <= row[1]), "last_date"] = row[1]
        group_number+=1
#%% Main
for group_number, df_grp in df_file.groupby("group_number"):
    print(group_number)
    # Loop for each gistda file
    for index, row in df_grp.iterrows():
        gdf = gpd.read_file(row["path"])
        
        # Check if gistda flood is within each strip_id
        for _, strip in df_sen1a_index.iterrows():
            if gdf.within(strip["geometry"]).any():
                strip_id = strip["SID"][1:]
                raster = rasterio.open(os.path.join(root_raster_temp, f"s1_valid_mask_{strip_id}.tif"))
                minx, miny, maxx, maxy = df_sen1a_template.loc[strip_id, ["minx", "miny", "maxx", "maxy"]]
                
                # Save shp -> rasterize
                path_temp_shp = os.path.join(root_temp, "temp.shp")
                path_temp_ras = os.path.join(root_temp, f"{strip_id}_{index}.tif")
                gdf[gdf.within(strip["geometry"])].to_file(path_temp_shp)
                
                gdal_rasterize(
                    path_temp_shp, path_temp_ras,
                    width=raster.width, height=raster.height,
                    outputBounds=(minx, miny, maxx, maxy),
                    outputType="Byte", allTouched=True,
                    burnValues=1
                )
                
                # Mask raster
                with rasterio.open(path_temp_ras) as src:
                    out_image, transform = mask(src, df_sen1a_index.loc[df_sen1a_index["SID"] == strip["SID"], "geometry"])
                    out_meta = src.meta.copy()
                    height, width = out_image[0].shape
                    out_meta.update({
                        'crs' : raster.crs,
                        'transform' : transform,
                        'width': width,
                        'height': height,
                        'dtype': src.dtypes[0]
                    })
                    with rasterio.open(path_temp_ras.replace(".tif", "_masked.tif"), "w", **out_meta) as dst:
                        dst.write(out_image)
        
    # After rasterized and masked -> Union raster
    for strip_id in np.unique([file[:3] for file in os.listdir(root_temp) if file.endswith("masked.tif")]):
        path_vrt = os.path.join(root_temp, f"temp_{strip_id}.vrt")
        create_vrt(
            path_save = path_vrt, 
            list_path_raster = [os.path.join(root_temp, file) for file in os.listdir(root_temp) if file.endswith("masked.tif") and file[:3] == strip_id],
            src_nodata=0, dst_nodata=0
        )
        raster = rasterio.open(path_vrt)
        raster_img = raster.read()
        raster_img = raster_img.sum(axis=0).astype("bool").astype("uint8")
        
        # Save final version
        path_final = os.path.join(root_final, f'{str(df_grp.iloc[0]["first_date"].date()).replace("-", "")}_{str(df_grp.iloc[0]["last_date"].date()).replace("-", "")}_S{strip_id}.tif')
        create_tiff(
            path_save = path_final,
            im = raster_img,
            projection = raster.crs.to_wkt(),
            geotransform = raster.get_transform(),
            drivername = "GTiff",
            dtype = "uint8"
        )
    
    # Clear all temp files
    del raster, raster_img
    for file in os.listdir(root_temp):
        try:
            os.remove(os.path.join(root_temp, file))
        except Exception as e:
            print(e)