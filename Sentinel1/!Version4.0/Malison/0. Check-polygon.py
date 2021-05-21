import os
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
tqdm.pandas()
#%%
path_s1_shp = r"F:\CROP-PIER\COPY-FROM-PIER\Sentinel1A_Index\Sentinel1_Index_4326.shp"
path_malison_shp = r"F:\CROP-PIER\CROP-WORK\vew_2020\gdf_tgia_checked_malison_20210407\gdf_tgia_checked_malison_20210407.shp"
path_thailand_shp = r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-province.shp"
#%%
gdf_s1 = gpd.read_file(path_s1_shp)
gdf_malison = gpd.read_file(path_malison_shp)
gdf_thailand = gpd.read_file(path_thailand_shp)

# For plot
gdf_s1['coords'] = gdf_s1['geometry'].apply(lambda x: x.representative_point().coords[:])
gdf_s1['coords'] = [coords[0] for coords in gdf_s1['coords']]

# Only s1b
gdf_s1 = gdf_s1.loc[gdf_s1["SID"].isin(['S302', 'S303', 'S304','S305', 'S306', 'S401', 'S402', 'S403'])]
s1b_polygon = gdf_s1.unary_union
#%%
list_within = []
for row in gdf_malison.itertuples():
    list_within.append(row.geometry.within(s1b_polygon))
gdf_malison = gdf_malison.assign(is_within = list_within)
#%%
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 11))
# Plot thailand and s1b
gdf_thailand.plot(ax=ax, facecolor="none", edgecolor="red")
gdf_s1.plot(ax=ax, facecolor="none", edgecolor="black")

# Label 
for idx, row in gdf_s1.iterrows():
    ax.annotate(s=row['SID'], xy=row['coords'], horizontalalignment='center', fontsize=10)

# Plot within and outside
gdf_malison[gdf_malison["is_within"]].centroid.plot(ax=ax, color="green", markersize=5, label=f"Within({gdf_malison['is_within'].sum()})")
gdf_malison[~gdf_malison["is_within"]].centroid.plot(ax=ax, color="blue", markersize=5, label=f"Outside({(~gdf_malison['is_within']).sum()})")
ax.legend(loc="best")
print(gdf_malison["is_within"].value_counts())
fig.savefig(r"F:\CROP-PIER\CROP-WORK\Presentation\20210520\Malison.png", bbox_inches="tight")
#%% Add PLANT_PROVINCE_CODE

df_p = gdf_thailand.geometry.progress_apply(lambda val: gdf_malison.within(val))
df_p = df_p.astype("uint8")
gdf_malison = gdf_malison.assign(PLANT_PROVINCE_CODE=[gdf_thailand.at[idx, "ADM1_PCODE"][2:] for idx in df_p.idxmax(axis=0)])
#%%
gdf_malison.to_file(path_malison_shp)

for p_code, group in gdf_malison.groupby("PLANT_PROVINCE_CODE"):
    path_shp = os.path.join(os.path.dirname(path_malison_shp), f"{p_code}.shp")
    group.to_file(path_shp)
