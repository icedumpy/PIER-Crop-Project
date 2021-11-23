import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
root_vew = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_pixval = r"F:/CROP-PIER/CROP-WORK/Sentinel1_dataframe_updated/hls_pixval_ndvi_16d_nr"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211118"
#%%
for file in os.listdir(root_pixval):
    p_code = file.split(".")[0][-2:]
    if p_code != "30":
        continue
    df_pixval = pd.read_parquet(os.path.join(root_pixval, file))
    df_vew = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if (file.split(".")[0][-7:-5] == p_code) and (file.split(".")[0][-4:] in ["2018", "2019", "2020"])], ignore_index=True)    
    print(p_code, len(df_vew), len(df_pixval["ext_act_id"].unique()))
#%%
df_pixval.loc[df_pixval["loss_ratio"] == 0, "label"] = 0
df_pixval.loc[(df_pixval["loss_ratio"] > 0) & (df_pixval["loss_ratio"] < 0.8), "label"] = 1
#%%
columns = [f"t{i:02d}" for i in range(1, 15)]
for rep_plant_date_16d, df_pixval_grp in df_pixval.groupby("rep_plant_date_16d"):
    plt.close("all")
    for column in columns:
        fig, ax = plt.subplots()
        print(int(column[1:]))
        
        sns.histplot(data=df_pixval_grp, x=column, common_norm=False,
                     hue="label", ax=ax, stat="probability")
        
        # df_pixval_grp.loc[df_pixval_grp["loss_ratio"] == 0, column].hist(bins="auto", ax=ax)
        # df_pixval_grp.loc[df_pixval_grp["loss_ratio"] != 0, column].hist(bins="auto", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"{rep_plant_date_16d.date()}\n{column}")
    break
#%%
df_normal = df_pixval_grp.loc[df_pixval_grp["loss_ratio"] == 0, columns]
df_loss   = df_pixval_grp.loc[df_pixval_grp["loss_ratio"] == 1, columns]
#%%
df_normal.median(axis=1).hist(bins="auto")
df_loss.median(axis=1).hist(bins="auto")
#%%
columns = [f"t{i:02d}" for i in range(5, 9)]

plt.figure()
df_normal.sample(n=len(df_loss))[columns].quantile(0.1, axis=1).hist(bins="auto", alpha=0.5, label="0")
df_loss[columns].quantile(0.1, axis=1).hist(bins="auto", alpha=0.5, label="1")
plt.legend()















