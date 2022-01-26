import os
import pandas as pd
#%%
df_drought = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\P'Tee+\df_y_test_4a1_rfc_wg_down25pct_5x_smote_sfm100_y_cls_drought_other.parquet")
df_drought = df_drought.reset_index()
df_flood   = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\P'Tee+\df_y_test_4a1_rfc_wg_down25pct_5x_smote_sfm100_y_cls_flood.parquet")
df_flood = df_flood.reset_index()
#%%
root_tai = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\Error\100k test"
for file in os.listdir(root_tai):
    df = pd.read_csv(os.path.join(root_tai, file)).iloc[:, 1:]
    df = df.rename(columns={
        "index_id":"ext_act_id", "predict":"y_pred",
        "p0":"prob_0",
        "p1":"prob_1",
        "p2":"prob_2",
    })    
    if file.endswith("dought.csv"):
        print(file, "drought")
        df = pd.merge(df, df_drought[["ext_act_id", "y_act"]], how="left", on="ext_act_id")
    elif file.endswith('flood.csv'):
        print(file, "flood")
        df = pd.merge(df, df_flood[["ext_act_id", "y_act"]], how="left", on="ext_act_id")
    df = df[df_drought.columns]
    df.to_parquet(os.path.join(root_tai, file.replace(".csv", ".parquet")))
#%%

