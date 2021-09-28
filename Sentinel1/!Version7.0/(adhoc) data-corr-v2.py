import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
def plot_corr_mat(corr_mat, figsize=(16, 9), square=True, annot=True, fmt=".2f", cbar=True):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_mat, square=True, annot=True, 
        fmt=".2f", cbar=False, ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig, ax
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\df_hls_sen1_v00_NE3.parquet")
df_datadict = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\data_dict_hls_sen1_v00.csv")
df_datadict.loc[df_datadict["source"].str.contains("NECTEC"), "source"] = "NECTEC"
columns = df.columns[7:-5]

root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210921\Fig"
os.makedirs(root_save, exist_ok=True)
#%% HLS NDVI
columns_hls_ph1 = df_datadict["column_nm"][(df_datadict["column_nm"].str.contains("ph1")) & (df_datadict["source"] == "HLS (NDVI)")].tolist()
columns_hls_ph2 = df_datadict["column_nm"][(df_datadict["column_nm"].str.contains("ph2")) & (df_datadict["source"] == "HLS (NDVI)")].tolist()
columns_hls_ph3 = df_datadict["column_nm"][(df_datadict["column_nm"].str.contains("ph3")) & (df_datadict["source"] == "HLS (NDVI)")].tolist()
columns_hls_ph4 = df_datadict["column_nm"][(df_datadict["column_nm"].str.contains("ph4")) & (df_datadict["source"] == "HLS (NDVI)")].tolist()

plt.close('all')
corr_mat = df[columns_hls_ph1].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) PH1")
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) PH1.png"), bbox_inches="tight")

corr_mat = df[columns_hls_ph2].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) PH2")
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) PH2.png"), bbox_inches="tight")

corr_mat = df[columns_hls_ph3].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) PH3")
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) PH3.png"), bbox_inches="tight")

corr_mat = df[columns_hls_ph4].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) PH4")
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) PH4.png"), bbox_inches="tight")

# My opition#1
columns_set1 = ["ph1_sum_dev_median", "ph2_sum_dev_median", 
                "ph3_sum_dev_median", "ph4_sum_dev_median",
                "wh_ssn_sum_dev_median"]
corr_mat = df[columns_set1].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) sum_dev_median")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) sum_dev_median.png"), bbox_inches="tight")

# My opition#1
columns_set2 = ["ph1_area_under_norm_index_median", "ph2_area_under_norm_index_median", 
                "ph3_area_under_norm_index_median", "ph4_area_under_norm_index_median",
                "wh_ssn_area_under_norm_index_median"]
corr_mat = df[columns_set2].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) area_under_norm_index_median")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) area_under_norm_index_median.png"), bbox_inches="tight")

# 
#%% S1
columns_set3 = ["dev_peak_p25","dev_peak_median", "dev_peak_p75"]
corr_mat = df[columns_set3].corr().abs()
fig, ax = plot_corr_mat(corr_mat)
ax.set_title("HLS (NDVI) dev_peak")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(root_save, "HLS (NDVI)", "HLS (NDVI) dev_peak.png"), bbox_inches="tight")















#%%
