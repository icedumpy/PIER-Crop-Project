import os
import pandas as pd
import geopandas as gpd
#%%
def p25(x):
    return x.quantile(0.25)
def p75(x):
    return x.quantile(0.75)
def ratio(x):
    return (x == 1).sum()/len(x)
def add_rice_area(df, df_tambon):
    # Calculate tambon area (sq. wa)
    gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp")
    gdf["tambon_pcode"] = gdf["ADM3_PCODE"].str.slice(2,).astype("int32")
    gdf = gdf.loc[gdf["tambon_pcode"].isin(df["tambon_pcode"])]
    gdf = gdf.to_crs({"init":'epsg:32647'})
    gdf["tambon_area_in_wa"] = gdf.geometry.area/4.0
    
    # Calculate mean percentage rice area (each year)
    temp = df.groupby(["final_plant_year", "tambon_pcode"]).agg({"total_actual_plant_area_in_wa":"sum"})
    temp = temp.reset_index()
    temp = pd.merge(temp, gdf[["tambon_pcode", "tambon_area_in_wa"]], how="left", on="tambon_pcode")
    temp["x_percent_rice_area"] = temp["total_actual_plant_area_in_wa"]/temp["tambon_area_in_wa"]
    
    # Calculate mean percentage rice area
    temp = temp.groupby("tambon_pcode").agg({"x_percent_rice_area":"mean"})
    temp = temp.reset_index()

    # Finish
    df_tambon = pd.merge(df_tambon, temp, how="left", on="tambon_pcode")
    return df_tambon
#%%
df_data_dict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_4a-20220127_1132-fix-errors.xlsx")
#%%
# Get list of all features
list_features = df_data_dict.loc[(df_data_dict["y_cls_drought_other"] == "Y") | (df_data_dict["y_cls_flood"] == "Y"), "column_nm"]

# Control features
dict_agg_control_features = {
    # This one is for y_label
    "y_cls_drought_other":"max",
    "y_cls_flood":"max",
    # Control features
    "x_dem_elevation":[p25, p75],
    "x_dem_gradient":[p25, p75],
    "total_actual_plant_area_in_wa":[p25, p75],
    "x_photo_sensitive_f":[ratio],
    "x_jasmine_rice_f":[ratio],
    "x_sticky_rice_f":[ratio],
    "x_rice_age_days":["mean"],
    "x_plant_info_v2_irrigation_f":[ratio]
}
#%%
# Create agg dict
dict_agg_features = {}
for feature in list_features:
    if not feature in dict_agg_control_features.keys():
        dict_agg_features[feature] = [p25, p75]
dict_agg_features = {
    **dict_agg_control_features,
    **dict_agg_features
}
#%%
# Load features dataframe
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_4a_NE3_compressed.parquet")
df.loc[df["y_cls_drought_other"] != 0, "y_cls_drought_other"] = 1
df.loc[df["y_cls_flood"] != 0, "y_cls_flood"] = 1

# Agg tambon label features
df_tambon = df.groupby(["final_plant_year", "tambon_pcode"]).agg(dict_agg_features)
df_tambon.columns = [df_tambon.columns.map("_".join)][0]
df_tambon = df_tambon[~df_tambon.iloc[:, 1:].isna().any(axis=1)]
df_tambon = df_tambon.reset_index()
df_tambon = df_tambon.rename(columns={
    "y_cls_drought_other_max":"y_cls_drought_other",
    "y_cls_flood_max":"y_cls_flood"
})
df_tambon = add_rice_area(df, df_tambon)
df_tambon.to_parquet(os.path.join(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Data-For-P-Tee", "df_pierxda_batch_4a_NE3_tambon_level.parquet"))
del df
#%%
# df_tambon = pd.read_parquet(os.path.join(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Data-For-P-Tee", "df_pierxda_batch_4a_NE3_tambon_level.parquet"))
#%%



















