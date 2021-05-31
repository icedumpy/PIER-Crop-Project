import os
import datetime
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
#%%
path_shp = r"F:\CROP-PIER\CROP-WORK\vew_2020\gdf_tgia_checked_malison_20210407\gdf_tgia_checked_malison_20210407.shp"
root_raster = r"C:\Users\PongporC\Desktop\temp\S1AB"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal_malison_2020"
#%%
gdf = gpd.read_file(path_shp, encoding="CP874")
#%%
for file_raster in os.listdir(root_raster):
    if file_raster.endswith(".vrt"):
        raster = rasterio.open(os.path.join(root_raster, file_raster))
        raster_descriptions = [datetime.datetime.strptime(name[:8] , "%Y%m%d") for name in raster.descriptions[2:]]
        strip_id = file_raster.split(".")[0][-3:]
        print(strip_id)
        list_df = []
        for _, row in gdf.iterrows():
            try:
                polygon = row.geometry
                plant_date = datetime.datetime.strptime(row["warranty_p"], "%d-%b-%Y")
                for index in range(0, len(raster_descriptions)-1):
                    if raster_descriptions[index] < plant_date <= raster_descriptions[index+1]:
                        break
                index_start = index+4
                index_stop = index+39
                if index_stop >= len(raster_descriptions)+4:
                    index_stop = len(raster_descriptions)+3
                
                masked, _ = mask(raster, [polygon], crop=True, nodata=-99, all_touched=False, indexes=list(np.r_[1, 2, range(index_start, index_stop)]))
                masked = masked.reshape(len(masked), -1).T
                masked = masked[(masked != -99).all(axis=1)]
                
                df = pd.DataFrame(len(masked)*[row])
                df["row"] = masked[:, 0].astype(int)
                df["col"] = masked[:, 1].astype(int)
                for i in range(2, masked.shape[1]):
                    df[f"t{i-2}"] = masked[:, i]
                list_df.append(df)
            except:
                continue
        
        if len(list_df) != 0:
            df = pd.concat(list_df, ignore_index=True)
            df = df.drop(columns="geometry")
        else:
            continue
        
        for p_code, df_grp in df.groupby("PLANT_PROV"):
            path_save = os.path.join(root_save, f"df_s1ab_temporal_p{p_code}_s{strip_id}.parquet")
            df_grp.to_parquet(path_save)
#%%
