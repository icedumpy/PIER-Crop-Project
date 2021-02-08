import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from icedumpy.io_tools import load_model
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020"
#%%
list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
#%%
for strip_id in list_strip_id[1::2]:
    print(strip_id)
    model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
    model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
    #%%
    # Load df_s1_temporal
    df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_s{strip_id}.parquet"))
    
    # Drop mis-located polygon
    df_s1_temporal = df_s1_temporal.loc[df_s1_temporal["is_within"]]
    
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
    df_vew = df_vew.loc[df_vew["ext_act_id"].isin(df_s1_temporal["ext_act_id"].unique())]
    df_vew["final_harvest_date"] = df_vew["final_plant_date"] + datetime.timedelta(days=180)
    
    # Merge df_vew into df_s1_temporal (date)
    df_s1_temporal = pd.merge(df_s1_temporal, df_vew[["ext_act_id", "final_plant_date", "final_harvest_date"]], on=["ext_act_id"], how="inner")
    # Reset column
    df_s1_temporal = df_s1_temporal[df_s1_temporal.columns[:7].tolist() + df_s1_temporal.columns[-2:].tolist() + df_s1_temporal.columns[7:-2].tolist()]
    
    # Drop if at the edge
    df_s1_temporal = df_s1_temporal.loc[~(df_s1_temporal.iloc[:, 9:] == 0).any(axis=1)]
    
    # Create df look up table (for df_s1_temporal column)
    df_content = pd.DataFrame(
        [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(9, len(df_s1_temporal.columns)-1)],
        columns=["index", "start", "stop"]
    )
    df_content = df_content.set_index("index")
    df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
    df_content = df_content.sort_index()
    
    # Group by p_code first
    for p_code, df_s1_temporal_p in df_s1_temporal.groupby(["p_code"]):
        list_df = []
        # Then group by ext_act_id
        for (ext_act_id, final_plant_date, final_harvest_date), df_s1_temporal_grp in tqdm(df_s1_temporal_p.groupby(["ext_act_id", "final_plant_date", "final_harvest_date"])):
            
            # Get column of final_plant_date and final_harvest_date
            column_plant = df_content.loc[((df_content["start"] < final_plant_date) & (df_content["stop"] >= final_plant_date))].index[0]
            try:
                column_harvest = df_content.loc[((df_content["start"] < final_harvest_date) & (df_content["stop"] >= final_harvest_date))].index[0]+2 # +2 for covering last date (+-2)
            except IndexError:
                # If harvest date is beyond lastest data
                if final_harvest_date > df_content.iat[-1, 1]:
                    column_harvest = df_content.index[-1]+1
            
            # Get data columns
            data = df_s1_temporal_grp.iloc[:, column_plant:column_harvest]
            
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
                data_model2 = np.vstack([data_model2.mean(axis=1), np.ones((data_model2.shape[0], )), list_age]).T
            model2_bin = model2.predict(data_model2)
            
            # Append data (np.int64, np.datetime64, int)
            df = pd.DataFrame({"ext_act_id":int(ext_act_id),
                               "date":list_date,
                               "predict":model2_bin.astype("int8"),
                             })
            list_df.append(df)
        
        # Create result dataframe and s ave
        df = pd.concat(list_df, ignore_index=True)
        path_save = os.path.join(root_save, f"s1_prediction_p{p_code}_s{strip_id}.parquet")
        df.to_parquet(path_save)
#%%