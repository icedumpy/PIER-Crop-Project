import os
import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from icedumpy.df_tools import set_index_for_loc
#%%
root_df_groundtruth = r"F:\CROP-PIER\CROP-WORK\doae-loss-data"
root_df_prediction_new = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020_Drop_FA"
#%%
path_df_groundtruth = r"F:\CROP-PIER\CROP-WORK\doae-loss-data\df_loss_data_2020_p30.parquet"
path_df_prediction_new = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020_Drop_FA\s1_prediction_p30_s304.parquet"
#%%
df_groundtruth = pd.read_parquet(path_df_groundtruth)
df_groundtruth = df_groundtruth.loc[(df_groundtruth["DANGER_TYPE_NAME"] == "nan") | (df_groundtruth["DANGER_TYPE_NAME"] == "อุทกภัย")]
df_prediction_new = pd.read_parquet(path_df_prediction_new)
df_prediction_new = set_index_for_loc(df_prediction_new, "ext_act_id")
#%%
list_groundtruth = []
list_pred = []
for row in tqdm(df_groundtruth.itertuples(), total=len(df_groundtruth)):
    ext_act_id = row.ext_act_id
    try:
        data = df_prediction_new.loc[ext_act_id]
    except KeyError:
        continue
    
    is_flood = row.DANGER_TYPE_NAME == "อุทกภัย"
    if is_flood:
        start_date = datetime.datetime.strptime(row.START_DATE, "%Y-%m-%d")
        # pred = (data.loc[data["date"] >= start_date, "predict"] > 0).any()
        pred = (data["predict"] > 0).any()
    else:
        pred = (data["predict"] > 0).any()
    list_groundtruth.append(is_flood)
    list_pred.append(pred)

cnf_matrix = confusion_matrix(list_groundtruth, list_pred)
detection_rate = cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])
false_alarm_rate = cnf_matrix[0, 1]/(cnf_matrix[0, 0]+cnf_matrix[0, 1])
print(f"Detection rate: {100*detection_rate:.2f} %")
print(f"False alarm rate:{100*false_alarm_rate:.2f} %")
