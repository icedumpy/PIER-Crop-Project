import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
path_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_flood_pixel_new\df_s1ab_flood_pixel_p45_s402.parquet"
path_nonflood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_nonflood_pixel_new\df_s1ab_nonflood_pixel_p45_s402.parquet"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210809"
columns = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
#%%
df_flood = pd.read_parquet(path_flood)
df_flood = df_flood[df_flood["loss_ratio"] >= 0.8]
df_flood["Label"] = "Flood"
df_nonflood = pd.read_parquet(path_nonflood)
df_nonflood["Label"] = "nonFlood"
df_nonflood = df_nonflood.sample(n=len(df_flood))

df = pd.concat([df_flood, df_nonflood], ignore_index=True)
df[columns] = 10*np.log10(df[columns])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df["Min(Diff)"] = df[columns].diff(axis=1).min(axis=1)
df["Min(Backscatter)"] = df[columns].min(axis=1)
#%%
plt.close('all')
plt.figure()
sns.histplot(data=df, x="Min(Diff)", hue="Label", stat="probability")
plt.savefig(os.path.join(root_save, "Min(Diff).png"), bbox_inches="tight")

plt.figure()
sns.histplot(data=df, x="Min(Backscatter)", hue="Label", stat="probability")
plt.savefig(os.path.join(root_save, "Min(Backscatter).png"), bbox_inches="tight")
#%%
