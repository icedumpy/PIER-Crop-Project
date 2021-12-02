import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#%%
root_features = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smrootzone_features"
#%%
columns = [
    'max',
    'pctl_max',
    'min',
    'pctl_min',
    'med',
    'pctl_med',
    'mean',
    'pctl_mean',
    'no_period_in_hist_p0-p10',
    'no_period_in_hist_p10-p20',
    'no_period_in_hist_p20-p30',
    'no_period_in_hist_p30-p40',
    'no_period_in_hist_p40-p50',
    'no_period_in_hist_p50-p60',
    'no_period_in_hist_p60-p70',
    'no_period_in_hist_p70-p80',
    'no_period_in_hist_p80-p90',
    'no_period_in_hist_p90-p100',
    'pct_period_in_hist_p0-p10',
    'pct_period_in_hist_p10-p20',
    'pct_period_in_hist_p20-p30',
    'pct_period_in_hist_p30-p40',
    'pct_period_in_hist_p40-p50',
    'pct_period_in_hist_p50-p60',
    'pct_period_in_hist_p60-p70',
    'pct_period_in_hist_p70-p80',
    'pct_period_in_hist_p80-p90',
    'pct_period_in_hist_p90-p100',
    'cnsct_period_under_5_strict',
    'cnsct_period_under_5_relax',
    'cnsct_period_under_10_strict',
    'cnsct_period_under_10_relax',
    'cnsct_period_under_15_strict',
    'cnsct_period_under_15_relax',
    'cnsct_period_under_20_strict',
    'cnsct_period_under_20_relax',
    'cnsct_period_above_80_strict',
    'cnsct_period_above_80_relax',
    'cnsct_period_above_85_strict',
    'cnsct_period_above_85_relax',
    'cnsct_period_above_90_strict',
    'cnsct_period_above_90_relax',
    'cnsct_period_above_95_strict',
    'cnsct_period_above_95_relax'
]
#%% Group by p_code & [2017, 2018, 2019, 2020]
list_info = []
for file in os.listdir(root_features):
    file = file
    list_info.append({
        "filename":file,
        "path":os.path.join(root_features, file),
        "p_code":file.split(".")[0].split("_")[-2],
        "year":file.split(".")[0].split("_")[-1],
    })
df_info = pd.DataFrame(list_info)
#%%
danger_type = "Drought"
for p_code, df_info_grp in df_info.groupby("p_code"):
    print(df_info_grp.filename)
    df_info_grp = df_info_grp[df_info_grp["year"].isin(["2017", "2018", "2019", "2020"])]
    df = pd.concat([pd.read_parquet(path) for path in df_info_grp.path], ignore_index=True)
    if (df["DANGER_TYPE_GRP"] == danger_type).sum():
        break
#%%
    plt.close("all")
    for column in columns:
        # Get data & drop defect
        df_loss = df.loc[df["DANGER_TYPE_GRP"] == danger_type, column]
        df_loss = df_loss[df_loss.between(df_loss.quantile(0.001), df_loss.quantile(0.999))]
        df_normal = df.loc[df["DANGER_TYPE_GRP"].isna(), column]
        df_normal = df_normal[df_normal.between(df_normal.quantile(0.001), df_normal.quantile(0.999))]
        arr_loss = df_loss.values
        arr_normal = df_normal.values

        # Calculate kde
        kde_loss = gaussian_kde(df_loss)
        kde_normal = gaussian_kde(df_normal)
        
        # Get kde
        x_min = min(kde_loss.dataset.min(), kde_normal.dataset.min())
        x_max = max(kde_loss.dataset.max(), kde_normal.dataset.max())
        x = np.linspace(x_min, x_max, 250)
        kde_loss = kde_loss(x)
        kde_normal = kde_normal(x)
        
        # Normalize kde
        y_min = min(kde_loss.min(), kde_normal.min())
        y_max = max(kde_loss.max(), kde_normal.max())
        kde_loss = (kde_loss-y_min)/(y_max-y_min)+1
        kde_normal = (kde_normal-y_min)/(y_max-y_min)+1
        
        # Calculate KL Divergence (P||Q + Q||P)/2
        kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)

        # Plot kde and show KL div
        plt.figure()
        plt.plot(x, kde_loss, label="Loss")
        plt.plot(x, kde_normal, label="Normal")
        plt.legend(loc=1)
        plt.title(f"{danger_type}\n{column}\nKL divergence: {kl_div:.4f}")

        print(f"{danger_type}, {column}: {kl_div:.4f}")
















