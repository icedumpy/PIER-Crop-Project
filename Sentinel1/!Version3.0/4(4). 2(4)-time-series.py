import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping, set_index_for_loc
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
strip_id = "304"
model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
#%%
# Load df mapping
df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]

# Load df vew
df_vew = load_vew(root_df_vew, list_p)
df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
df_vew = df_vew.loc[~(df_vew["ext_act_id"]%10).isin([8, 9])]
df_vew = df_vew[(df_vew["final_plant_date"] >= datetime.datetime(2018, 6, 1))]
df_vew = pd.merge(df_vew, df_mapping, how="inner", on=["new_polygon_id"])
df_vew["final_harvest_date"] = df_vew["final_plant_date"] + datetime.timedelta(days=180)

# Temp
df_vew = df_vew.loc[df_vew["PLANT_PROVINCE_CODE"] == 30]
df_vew = df_vew.loc[(df_vew["loss_ratio"] >= 0.8) | (df_vew["loss_ratio"] == 0.0)]
df_vew = df_vew.loc[df_vew["final_plant_date"].dt.year == 2019]

# load df s1_temporal
df_s1_temporal = pd.concat(
    [pd.read_parquet(os.path.join(root_df_s1_temporal, file))
     for file in os.listdir(root_df_s1_temporal) 
     if file.split(".")[0][-3:] == strip_id
     ], 
    ignore_index=True
)
df_s1_temporal = df_s1_temporal.loc[df_s1_temporal.new_polygon_id.isin(df_mapping.new_polygon_id.unique())]
df_s1_temporal.columns = [column[:8] if "_S1" in column else column for column in df_s1_temporal.columns]
df_s1_temporal = set_index_for_loc(df_s1_temporal, column="new_polygon_id")

# Create df look up table (for df_s1_temporal column)
df_content = pd.DataFrame(
    [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(7, len(df_s1_temporal.columns)-1)],
    columns=["index", "start", "stop"]
)
df_content = df_content.set_index("index")
df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
df_content = df_content.sort_index()
#%%
list_df = []
for (ext_act_id, new_polygon_id, final_plant_date, final_harvest_date), df_vewl_grp in tqdm(df_vew.groupby(["ext_act_id", "new_polygon_id", "final_plant_date", "final_harvest_date"])):
    try:
        column_plant = df_content.loc[((df_content["start"] < final_plant_date) & (df_content["stop"] >= final_plant_date))].index[0]
        try:
            column_harvest = df_content.loc[((df_content["start"] < final_harvest_date) & (df_content["stop"] >= final_harvest_date))].index[0]+2 # +2 for covering last date (+-2)
        except IndexError:
            # If harvest date is beyond lastest data
            if final_harvest_date > df_content.iat[-1, 1]:
                column_harvest = df_content.index[-1]+1
    except:
        continue
            
    data = df_s1_temporal.loc[new_polygon_id, df_s1_temporal.columns[column_plant:column_harvest]]
    if type(data) == pd.core.series.Series:
        data = data.to_frame().T
    
    # Sliding window (size = 4)
    list_age = []
    list_date = []
    list_data_model1 = []
    for i in range(len(data.columns)-3):
        date = datetime.datetime.strptime(data.columns[i+2], "%Y%m%d")
        age = (date - final_plant_date).days
        age = np.digitize(age, bins=[0, 40, 90], right=True)
        data_model1 = np.hstack([data.iloc[:, i:i+4].values, age+np.zeros((len(data), 1))]).reshape(-1, 5)
        list_age.append(age)
        list_date.append(date)
        list_data_model1.append(data_model1)
    
    # Predict model1 proba 
    data_model1 = np.vstack(list_data_model1)
    model1_prob = model1.predict_proba(data_model1)[:, -1]
    
    # Predict model2 
    data_model2 = np.vstack([model1_prob[i:i+len(data)] for i in range(0, len(model1_prob), len(data))])
    if len(data) > 1:
        data_model2 = np.vstack([data_model2.mean(axis=1), data_model2.std(axis=1), list_age]).T
    else:
        data_model2 = np.vstack([data_model2.mean(axis=1), np.zeros((data_model2.shape[0], )), list_age]).T
    model2_bin = model2.predict(data_model2)
    
    
    
    # Append data (np.int64, np.datetime64, int)
    df = pd.DataFrame({"ext_act_id": int(ext_act_id),
                       "date":list_date,
                       "predict":model2_bin.astype("int8"),
                       })
    list_df.append(df)
#%%
df
#%%

# Create result dataframe and save
df = pd.concat(list_df, ignore_index=True)
path_save = os.path.join(r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated", "s1_2019_prediction_p30_s304.parquet")
df.to_parquet(path_save)
