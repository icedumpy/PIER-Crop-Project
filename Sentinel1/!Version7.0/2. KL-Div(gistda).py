import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#%%
root_df = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_all_rice_by_year_plot_level_features(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211019\KL"
#%%
df = pd.concat([pd.read_parquet(os.path.join(root_df, file)) for file in os.listdir(root_df) if file.split(".")[0][-3:] == "402"], ignore_index=True)
#%%
df.loc[df["loss_ratio"] == 0, "y"] = 0
df.loc[df["loss_ratio"] != 0, "y"] = 1

columns = df.columns[-11:-2]
#%%
df.loc[df["DANGER_TYPE"].isin(["ฝนทิ้งช่วง", "ภัยแล้ง"]), "danger_type"] = "Drought"
df.loc[df["DANGER_TYPE"].isin(["อุทกภัย"]), "danger_type"] = "Flood"
df.loc[df["DANGER_TYPE"].isin(["ศัตรูพืชระบาด"]), "danger_type"] = "Other"
df.loc[df["DANGER_TYPE"].isna(), "danger_type"] = "Normal"
#%%
danger_type = "Flood"
dict_kl = dict()
dict_kl[danger_type] = dict()
os.makedirs(os.path.join(root_save, danger_type), exist_ok=True)
for column in columns:
    try:
        # Eliminate outlier of each class
        df_loss = df.loc[df["danger_type"] == danger_type, column]
        df_loss = df_loss[df_loss.between(df_loss.quantile(0.001), df_loss.quantile(0.999))]
        
        df_normal = df.loc[df["danger_type"].isna(), column]
        df_normal = df_normal[df_normal.between(df_normal.quantile(0.001), df_normal.quantile(0.999))]
        
        # Calculate kde
        kde_loss = gaussian_kde(df_loss)
        kde_normal = gaussian_kde(df_normal)
        
        # Get kde
        x_min = min(kde_loss.dataset.min(), kde_normal.dataset.min())
        x_max = max(kde_loss.dataset.max(), kde_normal.dataset.max())
        x = np.linspace(x_min, x_max, 200)
        kde_loss = kde_loss(x)
        kde_normal = kde_normal(x)
        
        # Normalize kde
        y_min = min(kde_loss.min(), kde_normal.min())
        y_max = max(kde_loss.max(), kde_normal.max())
        kde_loss = (kde_loss-y_min)/(y_max-y_min)+1
        kde_normal = (kde_normal-y_min)/(y_max-y_min)+1
        
        # Calculate KL Divergence
        kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)
        
        # Plot kde and show KL div
        plt.close("all")
        plt.figure()
        plt.plot(x, kde_loss, label="Loss")
        plt.plot(x, kde_normal, label="Normal")
        plt.legend(loc=1)
        plt.title(f"{danger_type}\n{column}\nKL divergence: {kl_div:.4f}")
        
        print(f"{danger_type}, {column}: {kl_div:.4f}")
        plt.savefig(os.path.join(root_save, danger_type, f"{column}.png"), bbox_inches="tight")
        
        # Save data
        dict_kl[danger_type][column] = kl_div
    except Exception as e:
        print(e)
        dict_kl[danger_type][column] = np.nan
#%%
df_kl = pd.DataFrame(dict_kl)
df_kl.to_csv(os.path.join(os.path.dirname(root_save), "KL.csv"))
#%%
import seaborn as sns
#%%
df = df[df["danger_type"].isin(["Flood", "Normal"])]
#%%
plt.close("all")
for column in columns:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, hue="danger_type", stat="probability",
                 binwidth=0.1, ax=ax)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
#%%
column = columns[3]
fig, ax = plt.subplots()
sns.histplot(data=df, x=column, hue="danger_type", stat="probability", ax=ax)
ax.set_xlim((0, 1))

#%%
columns = ['flood_ratio_1-5', 'flood_ratio_6-10', 'flood_ratio_11-15', 'flood_ratio_15+']
(df.loc[df["loss_ratio"] != 0, columns].sum(axis=1) == 0).sum()/len(df[df["loss_ratio"] != 0])
(df.loc[df["loss_ratio"] == 0, "flood_ratio_0"] == 1).sum()/len(df[df["loss_ratio"] == 0])


(df.loc[df["loss_ratio"] == 0, "flood_ratio_0"] == 1).sum()/len(df[df["loss_ratio"] == 0])
(df.loc[df["loss_ratio"] == 0, columns].sum(axis=1) == 0).sum()/len(df[df["loss_ratio"] == 0])
#%%
df["y_hat"] = (df[columns].sum(axis=1) != 0).astype("uint8")
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(df["y"], df["y_hat"])
#%%


(df.loc[df["loss_ratio"] == 0, columns].sum(axis=1) == 0).sum()/len(df[df["loss_ratio"] == 0])





























