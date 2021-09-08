import os
import utm
import numpy as np
import pandas as pd
from scipy.stats import mode
from icedumpy.geo_tools import convert_to_geodataframe
#%%
root = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
#%%
dct = dict()
for file in os.listdir(root):
    p_code = file.split(".")[0].split("_")[-2][1:]
    
    df = pd.read_parquet(os.path.join(root, file))
    gdf_4326 = convert_to_geodataframe(df, "polygon")
    
    # Check utm
    utm_zone = mode(((gdf_4326.centroid.x+180)/6).round().astype(int))[0][0]
    print(file, "|| UTM Zone:", utm_zone)

    # Convert to utm
    gdf_utm = gdf_4326.to_crs({"init":f"epsg:326{utm_zone}"})
    
    # Calculate const
    const = (gdf_utm.area/gdf_4326.area).to_list()
    
    if p_code not in dct.keys():
        dct[p_code] = const
    else:
        dct[p_code] = dct[p_code]+const
#%%
df = pd.DataFrame(columns=["p_code", "const"])
for p_code in dct.keys():
    df = df.append({"p_code":p_code, "const":np.mean(dct[p_code])}, ignore_index=True)
df.to_csv(r"C:\Users\PongporC\Desktop\RCT\4326-to-sqm-const.csv", index=False)
