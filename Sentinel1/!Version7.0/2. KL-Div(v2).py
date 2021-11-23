import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#%%
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210930\Fig"

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
#%%
dict_kl["All"] = dict()
os.makedirs(os.path.join(root_save, "All"), exist_ok=True)
for column in columns:
    # Eliminate outlier of each class
    df_loss = df.loc[df["y"] == 1, column]
    df_loss = df_loss[df_loss.between(df_loss.quantile(0.001), df_loss.quantile(0.999))]
    
    df_normal = df.loc[df["y"] == 0, column]
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
    plt.title(f"All\n{column}\nKL divergence: {kl_div:.4f}")
    
    print(f"All, {column}: {kl_div:.4f}")
    plt.savefig(os.path.join(root_save, "All", f"{column}.png"), bbox_inches="tight")
    
    # Save data
    dict_kl["All"][column] = kl_div
#%%
df_kl = pd.DataFrame(dict_kl)
df_kl.to_csv(os.path.join(os.path.dirname(root_save), "KL.csv"))
