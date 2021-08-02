import os
import pandas as pd
from tqdm import tqdm
#%%
root = r"C:\Users\PongporC\Desktop\RCT"
#%%
list_df = []
for file in os.listdir(root):
    if not file.startswith("Result"):
        list_df.append(pd.read_parquet(os.path.join(root, file)))
df = pd.concat(list_df, ignore_index=True)
#%%
list_dct = []
for ext_act_id, df_grp in tqdm(df.groupby("ext_act_id")):
    dct = df_grp.iloc[0, :3].to_dict()
    dct["predict"] = df_grp["predict"].mean()
    list_dct.append(dct)
df = pd.DataFrame(list_dct)
#%%
df['PLANT_PROVINCE_CODE'] = df["PAT"].str.slice(0, 2).astype("int")
df['PLANT_AMPHUR_CODE'] = df["PAT"].str.slice(2, 4).astype("int")
df['PLANT_TAMBON_CODE'] = df["PAT"].str.slice(4, 6).astype("int")
#%%
df.to_parquet(os.path.join(root, "Result.parquet"))
