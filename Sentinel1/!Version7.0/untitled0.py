import pandas as pd
import seaborn as sns
#%%
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\df_hls_sen1_v00_NE3.parquet")
#%%
df_corr = df.corr()
