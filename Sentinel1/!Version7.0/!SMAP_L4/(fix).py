import os
import pandas as pd
from tqdm import tqdm
#%%
dict_new_columns = { 
   'total_period':'total_sm_period',
   'cnsct_period_70_strict':'cnsct_sm_period_70_strict',
   'cnsct_period_70_relax' :'cnsct_sm_period_70_relax',
   'cnsct_period_80_strict':'cnsct_sm_period_80_strict',
   'cnsct_period_80_relax' :'cnsct_sm_period_80_relax',
   'cnsct_period_90_strict':'cnsct_sm_period_90_strict',
   'cnsct_period_90_relax' :'cnsct_sm_period_90_relax'
}
#%%
root = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smrootzone_features"
#%%
for file in tqdm(os.listdir(root)):
    path = os.path.join(root, file)
    df = pd.read_parquet(path)
    df = df.rename(columns=dict_new_columns)
    df.to_parquet(path)