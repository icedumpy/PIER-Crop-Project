import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_model
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping, set_index_for_loc
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
strip_id = "402"
model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
#%%
# Load df mapping
df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]

# Load df vew
df_vew = load_vew(root_df_vew, list_p)
df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
df_vew = df_vew[(df_vew["final_plant_date"] >= datetime.datetime(2018, 6, 1))]
df_vew = pd.merge(df_vew, df_mapping, how="inner", on=["new_polygon_id"])

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
df_vew_batch = df_vew.loc[df_vew["loss_ratio"] > 0].sample(n=100)
for row in df_vew_batch.itertuples():
    try:
        pred = [0, 0]
        new_polygon_id = row.new_polygon_id
        date_plant = row.final_plant_date
        date_harvest = date_plant+datetime.timedelta(days=180)
        date_flood = row.START_DATE
        
        column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
        column_harvest = df_content.loc[((df_content["start"] < date_harvest) & (df_content["stop"] >= date_harvest))].index[0]
        column_flood = df_content.loc[((df_content["start"] < date_flood) & (df_content["stop"] >= date_flood))].index[0]
        data = df_s1_temporal.loc[[new_polygon_id], df_s1_temporal.columns[column_plant:column_harvest+1]]
        for i in range(len(data.columns)-3):
            date = datetime.datetime.strptime(data.columns[i+2], "%Y%m%d")
            # print(date)
            # month = date.month
            age = (date - date_plant).days
            age = np.digitize(age, bins=[0, 40, 90], right=True)
            
            model1_prob = model1.predict_proba(np.hstack([data.iloc[:, i:i+4].values, age+np.zeros((len(data), 1))]).reshape(-1, 5))[:, -1]
            # model1_prob = model1.predict_proba(np.hstack([data.iloc[:, i:i+3].values, age+np.zeros((len(data), 1)), month+np.zeros((len(data), 1))]).reshape(-1, 5))[:, -1]
        
            model1_prob_mean = model1_prob.mean()
            if len(model1_prob) == 1:
                model1_prob_std = 0
            else:
                model1_prob_std = model1_prob.std()
            model2_bin = model2.predict(np.array([model1_prob_mean, model1_prob_std, age]).reshape(-1, 3))
            
            # print("---------------------------------")
            # print(data.columns[i:i+3].values, str(date_flood.date()).replace("-", ""), age)
            # print(data.iloc[:, i:i+3].values.mean(axis=0))
            # print(f"Model1(mean, std): ({model1_prob_mean:.2f}, {model1_prob_std:.2f})")
            # print(f"Model2(Bin): {int(model2_bin)}")
            # print()
            pred.append(model2_bin[0])
        while len(pred) < 31:
            pred.append(0.0)
        plt.close('all')
        fig, ax = plt.subplots()
        data = data.mean(axis=0)
        ax.plot([i for i in range(data.shape[-1])], data.values)
        for idx, value in enumerate(pred):
            if value != 0:
                ax.scatter(idx, data[idx], color="red")
                ax.text(idx, data[idx], f"{int(value)}")
        ax.axvline(column_flood - column_plant, color="red", linestyle="--")
        ax.grid()
        ax.set_title(f"{row.loss_ratio}")
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210208\Flood", f"{new_polygon_id}.png"))
    except IndexError:
        print(date_plant)
#%%
df_vew_batch = df_vew.loc[df_vew["loss_ratio"] == 0].sample(n=100)
for row in df_vew_batch.itertuples():
    try:
        pred = [0, 0]
        new_polygon_id = row.new_polygon_id
        date_plant = row.final_plant_date
        date_harvest = date_plant+datetime.timedelta(days=180)
        date_flood = row.START_DATE
        
        column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
        column_harvest = df_content.loc[((df_content["start"] < date_harvest) & (df_content["stop"] >= date_harvest))].index[0]
        # column_flood = df_content.loc[((df_content["start"] < date_flood) & (df_content["stop"] >= date_flood))].index[0]
        data = df_s1_temporal.loc[[new_polygon_id], df_s1_temporal.columns[column_plant:column_harvest+1]]
        for i in range(len(data.columns)-3):
            date = datetime.datetime.strptime(data.columns[i+2], "%Y%m%d")
            # print(date)
            # month = date.month
            age = (date - date_plant).days
            age = np.digitize(age, bins=[0, 40, 90], right=True)
            
            model1_prob = model1.predict_proba(np.hstack([data.iloc[:, i:i+4].values, age+np.zeros((len(data), 1))]).reshape(-1, 5))[:, -1]
            # model1_prob = model1.predict_proba(np.hstack([data.iloc[:, i:i+3].values, age+np.zeros((len(data), 1)), month+np.zeros((len(data), 1))]).reshape(-1, 5))[:, -1]
        
            model1_prob_mean = model1_prob.mean()
            if len(model1_prob) == 1:
                model1_prob_std = 0
            else:
                model1_prob_std = model1_prob.std()
            model2_bin = model2.predict(np.array([model1_prob_mean, model1_prob_std, age]).reshape(-1, 3))
            
            # print("---------------------------------")
            # print(data.columns[i:i+3].values, str(date_flood.date()).replace("-", ""), age)
            # print(data.iloc[:, i:i+3].values.mean(axis=0))
            # print(f"Model1(mean, std): ({model1_prob_mean:.2f}, {model1_prob_std:.2f})")
            # print(f"Model2(Bin): {int(model2_bin)}")
            # print()
            pred.append(model2_bin[0])
        while len(pred) < 31:
            pred.append(0.0)
        plt.close('all')
        fig, ax = plt.subplots()
        data = data.mean(axis=0)
        ax.plot([i for i in range(data.shape[-1])], data.values)
        for idx, value in enumerate(pred):
            if value != 0:
                ax.scatter(idx, data[idx], color="red")
                ax.text(idx, data[idx], f"{int(value)}")
        # ax.axvline(column_flood - column_plant, color="red", linestyle="--")
        ax.grid()
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210208\non-Flood", f"{new_polygon_id}.png"))
    except IndexError:
        print(date_plant)

