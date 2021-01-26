import os
import sys
sys.path.append(r"C:\Crop\code\pixel-value")
import re
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely import wkt
from pierpy import io_tools
tqdm.pandas()
#%%
root_loss = r"C:\Crop\data\Crop_Data\vew_polygon_id_plant_date_disaster"
filelist = [item for item in os.listdir(root_loss) if re.split('(\d+)', item.split(".")[0].split("_")[-2])[1] in ['2017', '2018']]
#%%
list_df = []
for filename in filelist:
    filepath = os.path.join(root_loss, filename)
    df = pd.read_parquet(filepath)
    list_df.append(df)
df = pd.concat(list_df)
#%%
df = df.drop(columns = ['PROFILE_CENTER_ID', 'ACTIVITY_ID', 'TYPE_CODE', 'DETAIL_CODE', 'BREED_CODE', 
                        'IRRIGATION_F', 'ACT_RAI_ORI', 'ACT_NGAN_ORI', 'ACT_WA_ORI', 'final_crop_year', 
                        'DANGER_TOTAL_RAI','DANGER_TOTAL_NGAN', 'DANGER_TOTAL_WA'])
df = df.reset_index(drop=True)
#%%
#df = df[df['PLANT_PROVINCE_CODE'].isin([60, 72])]
#%%
gdf = gpd.GeoDataFrame(df, crs={'init' : 'epsg:4326'})
gdf['geometry'] = gdf['final_polygon'].progress_apply(lambda val: wkt.loads(val))
gdf = gdf[gdf.geometry.type.isin(['MultiPolygon', 'Polygon'])]
gdf = gdf.drop(columns = ['final_polygon'])
#%%
gdf['loss_ratio'] = ""
gdf.loc[ pd.isnull(gdf['TOTAL_DANGER_AREA_IN_WA']), 'loss_ratio'] = 0
gdf.loc[~pd.isnull(gdf['TOTAL_DANGER_AREA_IN_WA']), 'loss_ratio'] = gdf['TOTAL_DANGER_AREA_IN_WA']/gdf['TOTAL_ACTUAL_PLANT_AREA_IN_WA']
#%%
gdf['START_DATE'] = gdf['START_DATE'].dt.date.astype('str')
#%%
#io_tools.write_pickle(gdf, r"C:\Crop\data\Processed Data\Loss-polygon\polygon.pkl")
gdf.to_file(r"C:\Crop\data\Processed Data\Loss-polygon\polygon.shp")