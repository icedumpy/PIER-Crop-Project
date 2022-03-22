.yimport os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
#%%
# Find max consecutive 1 (https://codereview.stackexchange.com/questions/138550/count-consecutive-ones-in-a-binary-list)
def len_iter(items):
    return sum(1 for _ in items)

def consecutive_one(data):
    return max(len_iter(run) for val, run in groupby(data) if val)
#%%
root_df_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\simple_index_smap_pixel_level_features_nrt(at-False)"
root_df_temporal =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smrootzone_temporal"
list_p = np.unique([file.split(".")[0][-7:-5] for file in os.listdir(root_df_temporal)]).tolist()
#%%
columns_all = [f"t{i}" for i in range(180)]
columns_s1 = [f"t{i}" for i in range(0, 40)]
columns_s2 = [f"t{i}" for i in range(40, 90)]
columns_s3 = [f"t{i}" for i in range(90, 120)]
columns_s4 = [f"t{i}" for i in range(120, 180)]
#%%
for file in os.listdir(root_df_temporal):
    print(file)
    path_file = os.path.join(root_df_temporal, file)
    df = pd.read_parquet(path_file)
    break