import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from icedumpy.io_tools import load_model
from icedumpy.df_tools import set_index_for_loc
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5_2020"
#%%
list_strip_id = ["302", "303", "304", "305", "401", "402", "403"]
for strip_id in list_strip_id:
    #%%
    model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
    model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
    #%%
    print(strip_id)
    model1 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model1.joblib")
    model2 = load_model(rf"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\{strip_id}-w-age\model2.joblib")
    
    # Load df_s1_temporal
    try:
        df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_s{strip_id}.parquet"))
        df_s1_temporal = set_index_for_loc(df_s1_temporal, column="ext_act_id")
    except FileNotFoundError:
        raise
    
    # Drop mis-located polygon
    df_s1_temporal = df_s1_temporal.loc[df_s1_temporal["is_within"]]
    
    # Drop tier 2
    df_s1_temporal = df_s1_temporal.loc[df_s1_temporal["tier"] == 1]
    
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
    df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
    df_vew = df_vew.assign(bin = np.digitize(df_vew.loss_ratio, bins=[0, .25, .50, .75], right=True))
    # Keep (non-flood and loss_ratio == 0 | flood and loss_ratio != 0)
    df_vew = df_vew.loc[((df_vew["DANGER_TYPE_NAME"].isna()) & (df_vew["loss_ratio"] == 0)) | (~((df_vew["loss_ratio"] == 0) & (df_vew["DANGER_TYPE_NAME"] == 'อุทกภัย')))]
    
    # Drop if at the edge
    df_s1_temporal = df_s1_temporal.loc[~(df_s1_temporal.iloc[:, 7:] == 0).any(axis=1)]
    
    # Create df look up table (for df_s1_temporal column)
    df_content = pd.DataFrame(
        [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(9, len(df_s1_temporal.columns)-1)],
        columns=["index", "start", "stop"]
    )
    df_content = df_content.set_index("index")
    df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
    df_content = df_content.sort_index()
    #%% Select only non-flood or loss_ratio >= 0.8
    df_vew = df_vew.loc[(df_vew["bin"] == 0) | (df_vew["loss_ratio"] >= 0.8)]
    # Down sampling non-flood
    df_vew = pd.concat([df_vew.loc[(df_vew["bin"] == 0)].sample(n=2*len(df_vew.loc[df_vew["bin"] != 0])), 
                        df_vew.loc[df_vew["bin"] != 0]])
    #%%
    list_gt = []
    list_data = []
    for row in tqdm(df_vew.itertuples(), total=len(df_vew)):
        try:
            # Find spot date
            if row.DANGER_TYPE_NAME == "อุทกภัย":
                date = row.START_DATE
            else:
                date = row.final_plant_date + datetime.timedelta(days=np.random.randint(12, 169))
            
            # Calculate age
            age = np.digitize((date - row.final_plant_date).days, bins=[0, 40, 90], right=True)
            
            # Find column's index
            column = df_content.loc[((df_content["start"] < date) & (df_content["stop"] >= date))].index[0]
            
            # Get data for model1
            data = df_s1_temporal.loc[row.ext_act_id, df_s1_temporal.columns[column-2:column+2]].values.reshape(-1, 4)
            data = np.hstack([data, age+np.zeros((len(data), 1))]).reshape(-1, 5)
            
            # Append data
            list_gt.append([row.bin] * len(data))
            list_data.append(data)
        except:
            continue
    #%%
    data = np.vstack(list_data)
    gt = np.array([j//4 for i in list_gt for j in i])
    model1_proba = model1.predict_proba(data)[:, -1]
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_roc_curve(model1, data, gt, label="2020", color="r-", ax=ax)
    ax = set_roc_plot_template(ax)
    ax.set_title(f'ROC Curve: {strip_id}\nAll_touched(False), Tier(1,)\nSamples: Flood:{(df_vew["bin"] == 4).sum():,}, Non-Flood:{(df_vew["bin"] == 0).sum():,}')
    fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210217\Fig\ROC", f"{strip_id}.png"))
