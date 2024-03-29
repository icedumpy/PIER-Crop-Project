import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
#%%
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210928\Fig"

df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\df_hls_sen1_v00_NE3.parquet")
df.loc[df["loss_ratio"] != 0, "y"] = 1

df_datadict = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\data_dict_hls_sen1_v00.csv")
df_datadict.loc[df_datadict["source"].str.contains("NECTEC"), "source"] = "NECTEC"
columns = df.columns[7:-9]
#%%
df.loc[df["danger_type"].isin(["ฝนทิ้งช่วง", "ภัยแล้ง"]), "danger_type"] = "Drought"
df.loc[df["danger_type"].isin(["อุทกภัย"]), "danger_type"] = "Flood"
df.loc[df["danger_type"].isin(["ศัตรูพืชระบาด"]), "danger_type"] = "Other"
#%%
dict_kl = dict()
for danger_type in df["danger_type"].unique():
    if danger_type == None:
        continue
    dict_kl[danger_type] = dict()
    os.makedirs(os.path.join(root_save, danger_type), exist_ok=True)
    for column in columns:
        print(f"{danger_type}: {column}")
        
        # Eliminate outlier of each class
        df_loss = df.loc[df["danger_type"] == danger_type, column]
        low_loss  = df_loss.quantile(0.001)
        high_loss = df_loss.quantile(0.999)
        df_loss = df_loss[df_loss.between(low_loss, high_loss)]
        
        df_normal = df.loc[df["danger_type"].isna(), column]
        low_normal  = df_normal.quantile(0.001)
        high_normal = df_normal.quantile(0.999)
        df_normal = df_normal[df_normal.between(low_normal, high_normal)]
        
        # Calculate kde
        kde_loss = gaussian_kde(df_loss)
        kde_normal = gaussian_kde(df_normal)
        
        # Get kl-divergence
        x_min = min(kde_loss.dataset.min(), kde_normal.dataset.min())
        x_max = min(kde_loss.dataset.max(), kde_normal.dataset.max())
        x = np.linspace(x_min, x_max, 100)
        kde_loss = kde_loss(x)
        kde_normal = kde_normal(x)
        kde_loss = np.where(kde_loss == 0, 1e-10, kde_loss)
        kde_normal = np.where(kde_normal == 0, 1e-10, kde_normal)
        kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)
        
        if kl_div == np.inf:
            print(f"{danger_type}: {column}: inf")
            continue
        
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
os.makedirs(os.path.join(root_save, "All"), exist_ok=True)
dict_kl["All"] = dict()
for column in columns:
    print(column)
    df_loss = df.loc[df["y"] == 1, column]
    low_loss  = df_loss.quantile(0.001)
    high_loss = df_loss.quantile(0.999)
    df_loss = df_loss[df_loss.between(low_loss, high_loss)]
    
    df_normal = df.loc[df["y"] == 0, column]
    low_normal  = df_normal.quantile(0.001)
    high_normal = df_normal.quantile(0.999)
    df_normal = df_normal[df_normal.between(low_normal, high_normal)]
    
    # Calculate kde
    kde_loss = gaussian_kde(df_loss)
    kde_normal = gaussian_kde(df_normal)
    
    # Get kl-divergence
    x_min = min(kde_loss.dataset.min(), kde_normal.dataset.min())
    x_max = min(kde_loss.dataset.max(), kde_normal.dataset.max())
    x = np.linspace(x_min, x_max, 100)
    kde_loss = kde_loss(x)
    kde_normal = kde_normal(x)
    kde_loss = np.where(kde_loss == 0, 1e-5, kde_loss)
    kde_normal = np.where(kde_normal == 0, 1e-5, kde_normal)
    kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)
            
    if kl_div == np.inf:
        print(f"{danger_type}: {column}: inf")
        continue
    
    # Plot
    plt.close("all")
    plt.figure()
    plt.plot(x, kde_loss, label="Loss")
    plt.plot(x, kde_normal, label="Normal")
    plt.legend(loc=1)
    plt.title(f"{column}\nKL divergence: {kl_div:.4f}")
    plt.savefig(os.path.join(root_save, "All", f"{column}.png"), bbox_inches="tight")
    
    # Save data
    dict_kl["All"][column] = kl_div
#%%
df_kl = pd.DataFrame(dict_kl)
