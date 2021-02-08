import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_mapping, set_index_for_loc
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
#%%
strip_id = "302"
model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
#%%
# Load df_s1_temporal
df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_s{strip_id}.parquet"))
df_s1_temporal.columns = [column[:8] if "_S1" in column else column for column in df_s1_temporal.columns]
list_p = df_s1_temporal["p_code"].unique().tolist()

# Load df vew
df_vew = pd.concat(
    [pd.read_parquet(os.path.join(root_df_vew, file)) 
     for file in os.listdir(root_df_vew) 
     if file.split(".")[0].split("_")[-1] in list_p
     ], 
    ignore_index=True
)

# Create df look up table (for df_s1_temporal column)
df_content = pd.DataFrame(
    [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(7, len(df_s1_temporal.columns)-1)],
    columns=["index", "start", "stop"]
)
df_content = df_content.set_index("index")
df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
df_content = df_content.sort_index()
#%%
for row in tqdm(df_vew.itertuples(), total=len(df_vew)):
    ext_act_id = row.ext_act_id
    row_col = df_mapping.loc[ext_act_id, "row_col"].values
    
    date_plant = row.final_plant_date
    date_harvest = row.final_harvest_date
    column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
    try:
        column_harvest = df_content.loc[((df_content["start"] < date_harvest) & (df_content["stop"] >= date_harvest))].index[0]
    except IndexError:
        # If harvest date is beyond lastest data
        if date_harvest > df_content.iat[-1, 1]:
            column_harvest = df_content.index[-1]
    
    # Get df temporal
    # try:
    data = df_s1_temporal.loc[row_col, df_s1_temporal.columns[column_plant:column_harvest+1]]
                
    # except:
        # continue
    pass
#%%
list_prediction = []
for ext_act_id, df_vew_grp in tqdm(df_vew.groupby(["ext_act_id"])):
    df_vew_grp = df_vew_grp.iloc[0] # idk why but it's faster than df_vew_grp.squeeze()
    row_col = df_mapping.loc[ext_act_id, ["row", "col"]]
    row_col = row_col["row"].astype(str) + "_" + row_col["col"].astype(str)

    date_plant = df_vew_grp.final_plant_date
    date_harvest = date_plant+datetime.timedelta(days=180)
    column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
    column_harvest = df_content.loc[((df_content["start"] < date_harvest) & (df_content["stop"] >= date_harvest))].index[0]
    try:
        data = df_s1_temporal.loc[[row_col], df_s1_temporal.columns[column_plant:column_harvest+1]]
    except:
        continue
    
    for i in range(len(data.columns)-3):
        date = datetime.datetime.strptime(data.columns[i+2], "%Y%m%d")
        age = (date - date_plant).days
        age = np.digitize(age, bins=[0, 40, 90], right=True)
        
        # Model1 prediction
        model1_prob = model1.predict_proba(np.hstack([data.iloc[:, i:i+4].values, age+np.zeros((len(data), 1))]).reshape(-1, 5))[:, -1]
        # Model1 prediction mean
        model1_prob_mean = model1_prob.mean()
        # Model1 prediction std
        if len(model1_prob) == 1:
            model1_prob_std = 0
        else:
            model1_prob_std = model1_prob.std()
            
        # Model2 prediction
        model2_bin = model2.predict(np.array([model1_prob_mean, model1_prob_std, age]).reshape(-1, 3))
        
        # Append data (np.int64, np.datetime64, int)
        prediction = (np.int64(ext_act_id), np.datetime64(date), int(model2_bin))
        list_prediction.append(prediction)
#%%
df_prediction = pd.DataFrame(list_prediction, columns=["ext_act_id", "date", "predict"])
#%%