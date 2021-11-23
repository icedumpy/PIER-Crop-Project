# 1 sort
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
root_df =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_temporal"
#%%
for file in tqdm(os.listdir(root_df)[4::5]):
    df = pd.read_parquet(os.path.join(root_df, file))
    df = df[df.columns[:34].tolist()+list(map(str, sorted(df.columns[34:].astype(int))))]
    df = df.rename(columns={i:f"t{i}" for i in df.columns[34:]})
    df.to_parquet(os.path.join(root_df, file))
#%% 2: missing columns
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
root_df =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_temporal"
columns_ideal = [f"t{i}" for i in range(26)]
#%%
for file in tqdm(os.listdir(root_df)):
    df = pd.read_parquet(os.path.join(root_df, file))
    for missing_column in [column for column in columns_ideal if not column in df.columns[34:]]:
        print(file, missing_column)
        df[missing_column] = np.nan
    df = df[df.columns[:34].tolist()+sorted(df.columns[34:], key=lambda val: int(val[1:]))]
    df.to_parquet(os.path.join(root_df, file))
#%% 3: Change ratio to percent
root_df = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\gistda_dri_features"
columns_pct_period_dri= [
    'pct_period_dri_in_hist_p0-p10' , 'pct_period_dri_in_hist_p10-p20', 'pct_period_dri_in_hist_p20-p30',
    'pct_period_dri_in_hist_p30-p40', 'pct_period_dri_in_hist_p40-p50', 'pct_period_dri_in_hist_p50-p60',
    'pct_period_dri_in_hist_p60-p70', 'pct_period_dri_in_hist_p70-p80', 'pct_period_dri_in_hist_p80-p90',
    'pct_period_dri_in_hist_p90-p100'
]
for file in tqdm(os.listdir(root_df)):
    df = pd.read_parquet(os.path.join(root_df, file))
    df[columns_pct_period_dri] = 100*df[columns_pct_period_dri]
    df.to_parquet(os.path.join(root_df, file))