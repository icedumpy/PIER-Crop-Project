import os
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
strip_id = "402"
df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]

df_vew = load_vew(root_df_vew, list_p)
df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
df_vew = df_vew[df_vew.loss_ratio > 0]
df_vew = df_vew[df_vew["START_DATE"].dt.year >= 2018]
df_vew = pd.merge(df_vew, df_mapping, how="inner", on=["new_polygon_id"])
#%%
df_vew_temp = df_vew.loc[df_vew["polygon_area_in_square_m"] >= 50000]
print(df_vew_temp.groupby(["ext_act_id"]).size().value_counts().sort_index())
#%%
df_vew_high_loss = df_vew_temp.loc[df_vew_temp["loss_ratio"] >= 0.8]
df_vew_med_loss = df_vew_temp.loc[(df_vew_temp["loss_ratio"] >= 0.5) & (df_vew_temp["loss_ratio"] < 0.8)]
df_vew_low_loss = df_vew_temp.loc[(df_vew_temp["loss_ratio"] >  0.0) & (df_vew_temp["loss_ratio"] < 0.5)]
#%%
high = df_vew_high_loss.loc[df_vew_high_loss["ext_act_id"] == df_vew_high_loss["ext_act_id"].sample(n=1).values[0]]
med = df_vew_med_loss.loc[df_vew_med_loss["ext_act_id"] == df_vew_med_loss["ext_act_id"].sample(n=1).values[0]]
low = df_vew_low_loss.loc[df_vew_low_loss["ext_act_id"] == df_vew_low_loss["ext_act_id"].sample(n=1).values[0]]
#%%
print(high.START_DATE.unique(), med.START_DATE.unique(), low.START_DATE.unique())
#%%
df_s1_temporal = pd.read_parquet(os.path.join(root_df_s1_temporal, f"df_s1ab_pixel_p{high['PLANT_PROVINCE_CODE'].unique()[0]}_s{strip_id}.parquet"))
#%%
fig, ax = plt.subplots()
df_s1_temporal.loc[df_s1_temporal.row.isin(high.row) & df_s1_temporal.col.isin(high.col)].iloc[:, 65:95].T.plot(legend=False, ax=ax)
fig, ax = plt.subplots()
df_s1_temporal.loc[df_s1_temporal.row.isin(med.row) & df_s1_temporal.col.isin(med.col)].iloc[:, 65:95].T.plot(legend=False, ax=ax)
fig, ax = plt.subplots()
df_s1_temporal.loc[df_s1_temporal.row.isin(low.row) & df_s1_temporal.col.isin(low.col)].iloc[:, 65:95].T.plot(legend=False, ax=ax)
#%%
import seaborn as sns
#%%
fig, ax = plt.subplots()
sns.lineplot(
    data=df_s1_temporal.loc[df_s1_temporal.row.isin(high.row) & df_s1_temporal.col.isin(high.col)].iloc[:, 65:95].melt(),
    x="variable", y="value",
    legend=False, ax=ax
)
fig, ax = plt.subplots()
sns.lineplot(
    data=df_s1_temporal.loc[df_s1_temporal.row.isin(med.row) & df_s1_temporal.col.isin(med.col)].iloc[:, 65:95].melt(),
    x="variable", y="value",
    legend=False, ax=ax
)
fig, ax = plt.subplots()
sns.lineplot(
    data=df_s1_temporal.loc[df_s1_temporal.row.isin(low.row) & df_s1_temporal.col.isin(low.col)].iloc[:, 65:95].melt(),
    x="variable", y="value",
    legend=False, ax=ax
)
#%%
# df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5\df_s1ab_pixel_p45_s402.parquet")
# #%%
# ice = df.loc[df["new_polygon_id"] == 6483103]

# ice.iloc[:, 7:].T.plot(legend=False)
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.df_tools import load_vew, clean_and_process_vew, load_mapping
#%%
root_df_vew = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v5(at-False)"
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_pixel_from_mapping_v5"
#%%
for strip_id in ["302", "303", "304", "305", "401", "402", "403"]:
    print(strip_id)
    df_mapping, list_p = load_mapping(root_df_mapping, strip_id = strip_id)
    df_mapping = df_mapping.loc[(df_mapping["tier"] == 1) & df_mapping["is_within"]]
    
    df_vew = load_vew(root_df_vew, list_p)
    df_vew = clean_and_process_vew(df_vew, df_mapping.new_polygon_id.unique())
    df_vew = df_vew[df_vew.loss_ratio > 0]
    df_vew = df_vew[df_vew["START_DATE"].dt.year >= 2018]
    
    plt.close("all")
    (df_vew["START_DATE"] - df_vew["final_plant_date"]).dt.days.hist(bins="auto")
    plt.savefig(os.path.join(r"F:\CROP-PIER\CROP-WORK\Presentation\20210120\Fig\start_date-plant_date", f"{strip_id}.png"))
































