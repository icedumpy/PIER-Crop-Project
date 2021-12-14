import pandas as pd
#%%
df_datadict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_3c.xlsx", engine='openpyxl')
df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
column_features = df_datadict.loc[df_datadict["training_feature_f"] == "Y", "column_nm"]
#%%
for (tambon_pcode, final_plant_year), df_grp in df.groupby(["tambon_pcode", "final_plant_year"]):
    print(tambon_pcode)
    print(df_grp.y.value_counts())
#%%
df_grp


#%%
tambon_pcode = 451101
final_plant_year = 2019
df_grp = df[(df["tambon_pcode"] == tambon_pcode) & (df["final_plant_year"] == final_plant_year)]
#%%
df_grp.y.value_counts()
#%%
temo = df_grp[column_features].describe().iloc[1:].melt(ignore_index=False)
temp["variable"] = temp["variable"] + "_" + temp.index 