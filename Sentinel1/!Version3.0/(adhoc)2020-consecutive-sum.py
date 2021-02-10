import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
#%%
root_df_prediction = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020(Malison)"
root_df_prediction_new = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_prediction_2020_Drop_FA(Malison)"
os.makedirs(root_df_prediction_new, exist_ok=True)
#%%
def worker(file):
    path_file = os.path.join(root_df_prediction, file)
    df_prediction = pd.read_parquet(path_file)
    list_df = []
    for ext_act_id, df_prediction_grp in df_prediction.groupby(["ext_act_id"]):
        arr = df_prediction_grp["predict"].values
        arr_new = [0]+[max(arr[i:i+2]) if (arr[i:i+2] > 0).all() else 0 for i in range(0, len(arr)-1, 1)]
        df_prediction_grp["predict"] = arr_new
        list_df.append(df_prediction_grp)
    df = pd.concat(list_df, ignore_index=True)
    df.to_parquet(os.path.join(root_df_prediction_new, file))
#%%
if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2) # Create a multiprocessing Pool
    pool.map(worker, os.listdir(root_df_prediction)) # Process data_inputs iterable with pool

