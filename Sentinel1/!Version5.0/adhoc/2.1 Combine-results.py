import os
import pandas as pd
import geopandas as gpd
#%%
root = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_for_report_20210625"
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp")
gdf.ADM3_PCODE = gdf.ADM3_PCODE.str.slice(2,)
#%%
df = pd.concat([pd.read_parquet(os.path.join(root, file)) for file in os.listdir(root) if file.endswith("parquet")], ignore_index=True)
df["strip_id"] = df["strip_id"].astype("uint")
df["count_pixel"] = 1
# df = df[df["loss_ratio"] >= 0.8]
df = df.groupby(["PAT", "strip_id"]).agg({**{column:"mean" for column in ['ext_act_id', 'loss_ratio', 'predict_proba', 'threshold', 'predict']}, **{"count_pixel":"sum"}})
df = df.reset_index()
list_df = []
for PAT, df_grp in df.groupby("PAT"):
    if len(df_grp) > 1:
        df_grp = pd.DataFrame(df_grp.loc[df_grp["predict"].idxmax()]).T
    list_df.append(df_grp)
df = pd.concat(list_df, ignore_index=True)
df["predict"]  = df["predict"].astype("float")
df = df.rename(columns={"PAT":"ADM3_PCODE"})
#%%
gdf = pd.merge(gdf, df[["ADM3_PCODE", "count_pixel", "predict"]], how="inner", on="ADM3_PCODE")
gdf = gdf.rename(columns={"predict":"DT"})
# gdf = gdf[["ADM3_PCODE", "count_pixel", "predict", "geometry"]]
gdf["count_pixel"] = gdf["count_pixel"].astype("uint")
gdf.to_file(os.path.join(root, "results.shp"))
#%%

