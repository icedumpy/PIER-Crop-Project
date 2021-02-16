import os
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
from icedumpy.df_tools import set_index_for_loc
from icedumpy.eval_tools import kappa_from_cnf_matrix
#%%
path_df_drone = r"F:\CROP-PIER\CROP-WORK\drone-area-ext_act_id-2020.csv"
path_df_loss_data = r"F:\CROP-PIER\CROP-WORK\doae-loss-data\df_loss_data_2020_p30.parquet"

path_df_prediction = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020\s1_prediction_p30_s304.parquet"
path_df_prediction_new = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020_Drop_FA\s1_prediction_p30_s304.parquet"
#%%
# Get ext_act_id
df_drone = pd.read_csv(path_df_drone)

# Get loss data
df_loss_data = pd.read_parquet(path_df_loss_data)
df_loss_data = df_loss_data.loc[df_loss_data["ext_act_id"].isin(df_drone["ext_act_id"])]

# Get prediction
df_prediction = pd.read_parquet(path_df_prediction)
df_prediction = df_prediction[df_prediction["ext_act_id"].isin(df_drone["ext_act_id"])]
df_prediction = set_index_for_loc(df_prediction, "ext_act_id")

# Get prediction (2 consecutive+)
df_prediction_new = pd.read_parquet(path_df_prediction_new)
df_prediction_new = df_prediction_new[df_prediction_new["ext_act_id"].isin(df_drone["ext_act_id"])]
df_prediction_new = set_index_for_loc(df_prediction_new, "ext_act_id")
#%%
# list_ext_act_id_FA = []
# list_groundtruth = []
# list_pred = []
# for row in df_loss_data.itertuples():
#     ext_act_id = row.ext_act_id
#     try:
#         data = df_prediction_new.loc[ext_act_id]
#     except KeyError:
#         continue
#     is_flood = row.DANGER_TYPE_NAME == "อุทกภัย"
#     if is_flood:
#         start_date = datetime.datetime.strptime(row.START_DATE, "%Y-%m-%d")
#         # pred = (data.loc[data["date"] >= start_date, "predict"] > 0).any()
#         pred = (data["predict"] > 0).any()
#     else:
#         pred = (data["predict"] > 0).any()
#     list_groundtruth.append(is_flood)
#     list_pred.append(pred)
    
#     # False alarm
#     if (not is_flood) and pred:
#         list_ext_act_id_FA.append(ext_act_id)        

# cnf_matrix = confusion_matrix(list_groundtruth, list_pred)
# print(kappa_from_cnf_matrix(cnf_matrix))
#%%
dict_result = dict()
for row in df_loss_data.itertuples():
    ext_act_id = row.ext_act_id
    try:
        data = df_prediction_new.loc[ext_act_id]
    except KeyError:
        continue
    pred = (data["predict"] > 0).any()
    groundtruth = row.DANGER_TYPE_NAME == "อุทกภัย"
    
    dict_result[ext_act_id] = str(int(groundtruth)) + str(int(pred))
df_result = pd.DataFrame.from_dict(dict_result, orient='index')
df_result = df_result.reset_index()
df_result.columns = ["ext_act_id", "result"]
#%%
from icedumpy.geo_tools import convert_to_geodataframe
df_polygon = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202\vew_polygon_id_plant_date_disaster_20210202_30.parquet")
df_polygon = df_polygon.loc[df_polygon["ext_act_id"].isin(df_loss_data.ext_act_id)]
df_polygon = pd.merge(df_polygon, df_result, on=["ext_act_id"], how="inner")
# df_polygon = df_polygon.loc[(df_polygon["PLANT_PROVINCE_CODE"] == 30) & (df_polygon["PLANT_AMPHUR_CODE"] == 14) & (df_polygon["PLANT_TAMBON_CODE"] == 7)]
df_polygon = df_polygon[["ext_act_id", "result", "final_plant_date", "final_polygon", "DANGER_TYPE_NAME"]]
df_polygon["final_plant_date"] = df_polygon["final_plant_date"].astype(str).str.split("-").str.join("-")
gdf = convert_to_geodataframe(df_polygon)
gdf.to_file(r"C:\Users\PongporC\Desktop\New folder\tmp.shp")
#%%
#%%
ext_act_id = 9240259842
data = df_prediction.loc[ext_act_id]



















