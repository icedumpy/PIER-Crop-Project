import os
import pandas as pd
from tqdm import tqdm
#%%
# Get plant_year
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
list_p = list(map(str, [30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 44, 45, 46, 48]))
list_df = []
for file in tqdm([file for file in os.listdir(root_vew) if (file.split(".")[0].split("_")[-2][1:] in list_p) and (int(file.split(".")[0].split("_")[-1]) >= 2017)]):
    df = pd.read_parquet(os.path.join(root_vew, file))
    list_df.append(df[["ext_act_id", "final_plant_year"]])
df_vew = pd.concat(list_df, ignore_index=True)
df_vew = df_vew.rename(columns={"final_plant_year":"plant_year"})
del list_df
#%%
df_p_tee = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\P'Tee+\df_y_test_4a1_rfc_wg_down25pct_5x_smote_sfm100_y_cls_drought_other.parquet")
df_p_tee = df_p_tee.reset_index()
df_tai = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\100k test\drf_dought.csv").iloc[:, 1:]
df_tai = df_tai.rename(columns={
    "index_id":"ext_act_id", "predict":"y_pred",
    "p0":"prob_0",
    "p1":"prob_1",
    "p2":"prob_2",
})
#%%
df_pred = pd.merge(df_tai, df_p_tee, how="left", on="ext_act_id", suffixes=("_P'Tee+", "_Tai"))
df_pred = pd.merge(df_pred, df_vew, how="left", on="ext_act_id")
#%%
df = df_pred.loc[(df_pred["plant_year"] == 2019), ["ext_act_id", "y_act", "y_pred_P'Tee+", "y_pred_Tai"]]
#%%
df.loc[df["ext_act_id"] == 9216843638]
