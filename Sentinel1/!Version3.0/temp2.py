import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from icedumpy.df_tools import load_vew, clean_and_process_vew, set_index_for_loc
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
for file in os.listdir(root_df_s1_temporal)[1::2]:
    try:
        p = file.split("pixel_p")[1][:2]
        scene = file.split("_s")[-1][:3]
        #%%
        df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, file))
        df_s1_temporal = df_s1_temporal.loc[(df_s1_temporal["tier"] == 1) & (df_s1_temporal["is_within"])]
        df_s1_temporal.columns = [column[:8] if "_S1" in column else column for column in df_s1_temporal.columns]
        df_s1_temporal = set_index_for_loc(df_s1_temporal, column="new_polygon_id")
        df_vew = clean_and_process_vew(load_vew(root_df_vew, [p]), df_s1_temporal.new_polygon_id.unique())
        df_vew = df_vew.loc[df_vew["final_plant_date"] >= datetime.datetime.strptime(df_s1_temporal.columns[7], "%Y%m%d")]
        df_content = pd.DataFrame(
            [(idx+1, datetime.datetime.strptime(df_s1_temporal.columns[idx], "%Y%m%d"), datetime.datetime.strptime(df_s1_temporal.columns[idx+1], "%Y%m%d")) for idx in range(7, len(df_s1_temporal.columns)-1)],
            columns=["index", "start", "stop"]
        )
        df_content = df_content.set_index("index")
        df_content.loc[df_content.index[0]-1, ["start", "stop"]] = [df_content.loc[df_content.index[0], "start"]-datetime.timedelta(days=6), df_content.loc[df_content.index[0], "start"]]
        df_content = df_content.sort_index()
        #%%
        list_s1_temporal_normal = []
        list_s1_temporal_flood = []
        list_s1_flood_index = []
        for row in tqdm(df_vew.itertuples(), total=len(df_vew)):
            new_polygon_id = row.new_polygon_id
            date_plant = row.final_plant_date
            date_harvest = date_plant+datetime.timedelta(days=180)
            column_plant = df_content.loc[((df_content["start"] < date_plant) & (df_content["stop"] >= date_plant))].index[0]
            column_harvest = df_content.loc[((df_content["start"] < date_harvest) & (df_content["stop"] >= date_harvest))].index[0]
            
            s1_temporal = df_s1_temporal.loc[new_polygon_id, df_s1_temporal.columns[column_plant:column_harvest+1]].values
            if s1_temporal.shape[-1] == 31:
                if row.DANGER_TYPE_NAME == None:
                    list_s1_temporal_normal.append(s1_temporal)
                elif row.DANGER_TYPE_NAME in ["ภัยแล้ง", "ฝนทิ้งช่วง"]:
                    column_flood = df_content.loc[((df_content["start"] < row.START_DATE) & (df_content["stop"] >= row.START_DATE))].index[0]
                    list_s1_temporal_flood.append(s1_temporal)
                    
                    if len(s1_temporal.shape) == 2:
                        list_s1_flood_index.append([column_flood-column_plant]*s1_temporal.shape[0])
                    else:
                        list_s1_flood_index.append([column_flood-column_plant])
        #%%
        df_s1_temporal_normal = pd.DataFrame(np.vstack(list_s1_temporal_normal)).astype("float32")
        if len(list_s1_temporal_flood) != 0:
            df_s1_temporal_flood = pd.DataFrame(np.vstack(list_s1_temporal_flood)).astype("float32")
        else:
            df_s1_temporal_flood = pd.DataFrame()
        # df_s1_temporal_flood["column_flood"] = np.concatenate(list_s1_flood_index)
        df_s1_temporal_normal["Disaster"] = "Normal"
        df_s1_temporal_flood["Disaster"] = "Drought"
        df_s1_temporal = pd.concat([df_s1_temporal_normal, df_s1_temporal_flood])
        #%%
        plt.close('all')
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.lineplot(
            data=df_s1_temporal.melt(id_vars="Disaster"),
            x="variable", y="value", hue="Disaster",
            ax=ax
        )
        ax.set_xticks(np.arange(0, 31, 2))
        ax.grid()
        ax.set_xlabel("Time")
        ax.set_title(f"Scene_P: {scene}_{p}")
        fig.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210113\Fig_temporal_drought", f"{scene}_{p}.png"))
    except Exception as e:
        print(file, e)
#%%
df_s1_temporal.loc[df_s1_temporal["new_polygon_id"].isin(df_vew.loc[~df_vew["DANGER_TYPE_NAME"].isna(), "new_polygon_id"])].mean(axis=0)[7:].plot()
