import os
import pandas as pd
from tqdm import tqdm
#%%
root = r"C:\Users\PongporC\Desktop\RCT"
#%%
list_df = []
for file in os.listdir(root):
    if (not file.startswith("Result")) and file.endswith(".parquet"):
        list_df.append(pd.read_parquet(os.path.join(root, file)))
df = pd.concat(list_df, ignore_index=True)
#%%
list_dct = []
for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
    dct = df_grp.iloc[0, :3].to_dict()
    dct["threshold"] = df_grp.iloc[0]["threshold"]
    dct["predicted_loss_ratio"] = df_grp["predict"].mean()
    dct["prob_min"] = df_grp["predict_proba"].min()
    dct["prob_max"] = df_grp["predict_proba"].max()
    dct["prob_p25"] = df_grp["predict_proba"].quantile(0.25)
    dct["prob_p50"] = df_grp["predict_proba"].quantile(0.5)
    dct["prob_p75"] = df_grp["predict_proba"].quantile(0.75)
    list_dct.append(dct)
df = pd.DataFrame(list_dct)
#%%
df['PLANT_PROVINCE_CODE'] = df["PAT"].str.slice(0, 2).astype("int")
df['PLANT_AMPHUR_CODE'] = df["PAT"].str.slice(2, 4).astype("int")
df['PLANT_TAMBON_CODE'] = df["PAT"].str.slice(4, 6).astype("int")
#%%
df.to_parquet(os.path.join(root, "Result.parquet"))
