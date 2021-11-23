import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\df_hls_sen1_v00_NE3.parquet")
df_datadict = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\data_dict_hls_sen1_v00.csv")
df_datadict.loc[df_datadict["source"].str.contains("NECTEC"), "source"] = "NECTEC"
columns = df.columns[7:-5]

root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210921\Fig"
os.makedirs(root_save, exist_ok=True)
#%% Sorted corr
corr_mat  = df.corr()
corr_mat  = corr_mat.abs()
corr_mat  = corr_mat.loc[columns, columns]

# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat = corr_mat.where(
    np.triu(
        np.ones(corr_mat.shape), k=1
    ).astype('bool')
)
  
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
  
# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values(ascending=False)
sorted_mat = sorted_mat.reset_index()
sorted_mat.columns = ["Feature#1", "Feature#2", "Correlation"]
sorted_mat.to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20210921\features_corr.csv", index=False)
#%% Group by source
for source, df_datadict_grp in df_datadict.groupby("source"):
    if (source == "DOAE") or (source == "k-means clustering"):
        continue
    os.makedirs(os.path.join(root_save, source), exist_ok=True)
    
    print(source)
    columns_grp = df_datadict_grp["column_nm"].tolist()
    corr_mat = df[columns_grp].corr().abs()
    ##
    ##
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(corr_mat, square=False, annot=True,
                fmt=".2f", cbar=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(root_save, f"{source}.png"), bbox_inches="tight")
#%%
for feature, corr_mat_grp in corr_mat.iterrows():
    print(feature)
    corr_mat_grp = corr_mat_grp.drop(feature)
    corr_mat_grp = corr_mat_grp.sort_values(ascending=False).head(10)
    
    plt.close('all')
    fig, ax = plt.subplots()
    sns.heatmap(
        corr_mat_grp.values.reshape(1, -1), 
        square=True, annot=True, cbar=False, ax=ax
    )
    ax.set_xticklabels(corr_mat_grp.index.tolist(), rotation=90)
    ax.set_yticklabels([feature], rotation=0)
    fig.tight_layout()
    ax.set_title(f"Source: {source}")
    fig.savefig(os.path.join(root_save, source, f"{feature}.png"), bbox_inches="tight")
