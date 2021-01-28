import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
root_df_s1_temporal = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_temporal"
#%%
# [(column, 6*int(column[1:])) for column in columns]
columns = [f"t{i}" for i in range(31)]
age_group1 = [f"t{i}" for i in range(0, 7)]
age_group2 = [f"t{i}" for i in range(6, 15)]
age_group3 = [f"t{i}" for i in range(14, 20)]
age_group4 = [f"t{i}" for i in range(19, 31)]
#%%
df_s1_temporal = pd.concat([pd.read_parquet(os.path.join(root_df_s1_temporal, file)) for file in os.listdir(root_df_s1_temporal) if file.split(".")[0][-3:]], ignore_index=True)
#%%
fig, ax = plt.subplots()
ax.plot(age_group1, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] == 0.0, age_group1].mean(axis=0))
ax.plot(age_group2, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] == 0.0, age_group2].mean(axis=0))
ax.plot(age_group3, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] == 0.0, age_group3].mean(axis=0))
ax.plot(age_group4, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] == 0.0, age_group4].mean(axis=0))

ax.plot(age_group1, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] >= 0.8, age_group1].mean(axis=0), "--")
ax.plot(age_group2, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] >= 0.8, age_group2].mean(axis=0), "--")
ax.plot(age_group3, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] >= 0.8, age_group3].mean(axis=0), "--")
ax.plot(age_group4, df_s1_temporal.loc[df_s1_temporal["loss_ratio"] >= 0.8, age_group4].mean(axis=0), "--")
#%%
ax1.