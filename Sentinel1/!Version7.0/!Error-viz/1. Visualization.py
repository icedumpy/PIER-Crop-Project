import os
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from icedumpy.geo_tools import convert_to_geodataframe
#%%
root_save = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\SHP"
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_p_tee = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\P'Tee+"
#%% Get ext_act and polygon
list_p = list(map(str, [30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 44, 45, 46, 48]))
list_df = []
for file in tqdm([file for file in os.listdir(root_vew) if (file.split(".")[0].split("_")[-2][1:] in list_p) and (int(file.split(".")[0].split("_")[-1]) >= 2017)]):
    df = pd.read_parquet(os.path.join(root_vew, file))
    list_df.append(df[["ext_act_id", "polygon"]])
df = pd.concat(list_df, ignore_index=True)
gdf = convert_to_geodataframe(df, polygon_column="polygon")
#%% Loop for each result file
for file in os.listdir(root_p_tee):
    print(file)
    path_save = os.path.join(root_save, file.replace(".parquet", ".shp"))
    if os.path.exists(path_save):
        continue
    
    path = os.path.join(root_p_tee, file)
    df = pd.read_parquet(path)
    df = df.reset_index()
    
    # Merge polygon 
    df = pd.merge(df, gdf, how="left", on="ext_act_id")
    df = gpd.GeoDataFrame(df)
    
    # Assign result type
    df.loc[(df["y_pred"] == 0) & (df["y_act"] == 0), "resul_type"] = 1
    df.loc[(df["y_pred"] == 1) & (df["y_act"] == 1), "resul_type"] = 2
    df.loc[(df["y_pred"] == 2) & (df["y_act"] == 2), "resul_type"] = 3
    df.loc[df["resul_type"] == 1, "[Pred Act]"] = "1:[0 0]"
    df.loc[df["resul_type"] == 2, "[Pred Act]"] = "2:[1 1]"
    df.loc[df["resul_type"] == 3, "[Pred Act]"] = "3:[2 2]"
    
    # Error type: Over (pred-act)
    # 4: 2-0
    # 5: 1-0
    # 6: 2-1
    df.loc[(df["y_pred"] == 2) & (df["y_act"] == 0), "resul_type"] = 4
    df.loc[(df["y_pred"] == 1) & (df["y_act"] == 0), "resul_type"] = 5
    df.loc[(df["y_pred"] == 2) & (df["y_act"] == 1), "resul_type"] = 6
    df.loc[df["resul_type"] == 4, "[Pred Act]"] = "4:[2 0]"
    df.loc[df["resul_type"] == 5, "[Pred Act]"] = "5:[1 0]"
    df.loc[df["resul_type"] == 6, "[Pred Act]"] = "6:[2 1]"
    
    # Error type: Under (pred-act)
    # 7: 0-2
    # 8: 0-1
    # 9: 1-2
    df.loc[(df["y_pred"] == 0) & (df["y_act"] == 2), "resul_type"] = 7
    df.loc[(df["y_pred"] == 0) & (df["y_act"] == 1), "resul_type"] = 8
    df.loc[(df["y_pred"] == 1) & (df["y_act"] == 2), "resul_type"] = 9
    df.loc[df["resul_type"] == 7, "[Pred Act]"] = "7:[0 2]"
    df.loc[df["resul_type"] == 8, "[Pred Act]"] = "8:[0 1]"
    df.loc[df["resul_type"] == 9, "[Pred Act]"] = "9:[1 2]"
    
    # Save gdf
    df.to_file(path_save)
#%%
root_tai = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\100k test"
for file in os.listdir(root_tai):
    print(file)
    path_save = os.path.join(root_save, file.replace(".csv", ".shp"))
    if os.path.exists(path_save):
        continue
    
    path = os.path.join(root_tai, file)
    df_tai = pd.read_csv(path).iloc[:, 1:]
    df_tai = df_tai.rename(columns={"index_id":"ext_act_id", "predict":"y_pred"})
    
    # Merge 
    df_tai = pd.merge(df_tai, df[["ext_act_id", "y_act"]], how="left", on="ext_act_id")
    
    # Merge polygon 
    df_tai = pd.merge(df_tai, gdf, how="left", on="ext_act_id")
    df_tai = gpd.GeoDataFrame(df_tai)
    
    # Assign result type
    df_tai.loc[(df_tai["y_pred"] == 0) & (df_tai["y_act"] == 0), "resul_type"] = 1
    df_tai.loc[(df_tai["y_pred"] == 1) & (df_tai["y_act"] == 1), "resul_type"] = 2
    df_tai.loc[(df_tai["y_pred"] == 2) & (df_tai["y_act"] == 2), "resul_type"] = 3
    df_tai.loc[df_tai["resul_type"] == 1, "[Pred Act]"] = "1:[0 0]"
    df_tai.loc[df_tai["resul_type"] == 2, "[Pred Act]"] = "2:[1 1]"
    df_tai.loc[df_tai["resul_type"] == 3, "[Pred Act]"] = "3:[2 2]"
    
    # Error type: Over (pred-act)
    # 4: 2-0
    # 5: 1-0
    # 6: 2-1
    df_tai.loc[(df_tai["y_pred"] == 2) & (df_tai["y_act"] == 0), "resul_type"] = 4
    df_tai.loc[(df_tai["y_pred"] == 1) & (df_tai["y_act"] == 0), "resul_type"] = 5
    df_tai.loc[(df["y_pred"] == 2) & (df_tai["y_act"] == 1), "resul_type"] = 6
    df_tai.loc[df_tai["resul_type"] == 4, "[Pred Act]"] = "4:[2 0]"
    df_tai.loc[df_tai["resul_type"] == 5, "[Pred Act]"] = "5:[1 0]"
    df_tai.loc[df_tai["resul_type"] == 6, "[Pred Act]"] = "6:[2 1]"
    
    # Error type: Under (pred-act)
    # 7: 0-2
    # 8: 0-1
    # 9: 1-2
    df_tai.loc[(df_tai["y_pred"] == 0) & (df_tai["y_act"] == 2), "resul_type"] = 7
    df_tai.loc[(df_tai["y_pred"] == 0) & (df_tai["y_act"] == 1), "resul_type"] = 8
    df_tai.loc[(df_tai["y_pred"] == 1) & (df_tai["y_act"] == 2), "resul_type"] = 9
    df_tai.loc[df_tai["resul_type"] == 7, "[Pred Act]"] = "7:[0 2]"
    df_tai.loc[df_tai["resul_type"] == 8, "[Pred Act]"] = "8:[0 1]"
    df_tai.loc[df_tai["resul_type"] == 9, "[Pred Act]"] = "9:[1 2]"
    
    # Save gdf
    df_tai.to_file(path_save)
    break

















