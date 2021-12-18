import pandas as pd
from tqdm import tqdm
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.parquet")
df = df[df["y"].isin([0, 3, 4])]
df.loc[df["y"] != 0, "y"] = 1
#%%
# Loop for each (year, tambon)
for (final_plant_year, tambon_pcode), df_grp in tqdm(df.groupby(["final_plant_year", "tambon_pcode"])):
    # Label data as normal or flood
    y = df_grp["y"].unique().max() # if [0] then 0, elif [0, 1] then 1
    



    break
#%%
[column for column in df.columns if "s1" in column]
df_grp["x_smap_soil_moist_pctl_max_sm"].max()
