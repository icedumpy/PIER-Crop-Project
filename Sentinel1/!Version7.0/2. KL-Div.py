import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
#%%
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210927\Fig"

df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\df_hls_sen1_v00_NE3.parquet")
df.loc[df["loss_ratio"] == 0, "y"] = 1
df.loc[df["loss_ratio"] != 0, "y"] = 2

df_datadict = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\data_dict_hls_sen1_v00.csv")
df_datadict.loc[df_datadict["source"].str.contains("NECTEC"), "source"] = "NECTEC"
columns = df.columns[7:-5]
#%%
df.loc[df["danger_type"].isin(["ฝนทิ้งช่วง", "ภัยแล้ง"]), "danger_type"] = "Drought"
df.loc[df["danger_type"].isin(["อุทกภัย"]), "danger_type"] = "Flood"
df.loc[df["danger_type"].isin(["ศัตรูพืชระบาด"]), "danger_type"] = "Other"
#%%
danger_type = "อุทกภัย"
column = "bc(t)_min"
column = "ph1_mean_rainfall"
#%%
dict_kl = dict()
for danger_type in df["danger_type"].unique():
    if danger_type == None:
        continue
    dict_kl[danger_type] = dict()
    os.makedirs(os.path.join(root_save, danger_type), exist_ok=True)
    for column in columns:
        print(f"{danger_type}: {column}")
        
        # Calculate kde
        kde_loss = gaussian_kde(df.loc[df["danger_type"] == danger_type, column])
        kde_normal = gaussian_kde(df.loc[df["danger_type"].isna(), column])
        
        # Get kl-divergence
        x_min = min(np.quantile(kde_loss.dataset, 0.01), np.quantile(kde_normal.dataset, 0.01))
        x_max = max(np.quantile(kde_loss.dataset, 0.99), np.quantile(kde_normal.dataset, 0.99))
        x = np.linspace(x_min, x_max, 200)
        kde_loss = kde_loss(x)
        kde_normal = kde_normal(x)
        kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)
        
        # Plot
        plt.close("all")
        plt.figure()
        plt.plot(x, kde_loss, label="Loss")
        plt.plot(x, kde_normal, label="Normal")
        plt.legend(loc=1)
        plt.title(f"{danger_type}\n{column}\nKL divergence: {kl_div:.4f}")
        plt.savefig(os.path.join(root_save, danger_type, f"{column}.png"), bbox_inches="tight")
        
        # Save data
        dict_kl[danger_type][column] = kl_div
#%%
df_kl = pd.DataFrame([dict_kl]).T
