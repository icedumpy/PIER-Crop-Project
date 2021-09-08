import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icedumpy.geo_tools import convert_to_geodataframe
#%%
root = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
#%% 
list_area_reported = []
list_area_polygon = []
for file in os.listdir(root):
    print(file)
    path_file = os.path.join(root, file)
    df = pd.read_parquet(path_file)
    gdf = convert_to_geodataframe(df, 'polygon', to_crs={"init":'epsg:32647'})
    list_area_reported.append(4*gdf.TOTAL_ACTUAL_PLANT_AREA_IN_WA)
    list_area_polygon.append(gdf.area)
#%%
a = [item2 for item1 in list_area_reported for item2 in item1.to_list()]
b = [item2 for item1 in list_area_polygon for item2 in item1.to_list()]
df = pd.DataFrame(np.vstack([a, b])).T
df.columns = ["reported", "polygon"]
#%%
df_downsam = df.sample(frac=0.01)
df_downsam = df_downsam[df_downsam["reported"] <= df_downsam["reported"].quantile(0.95)]
#%%
plt.close("")
sns.histplot(data=df_downsam, x="reported", 
             binrange=(0, df_downsam["reported"].quantile(0.95)), 
             bins="auto", kde=True)
#%%
