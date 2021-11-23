import os
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

def extract_extreme_features(df_temporal, columns):
    df_temporal = df_temporal.copy()
    list_df = []
    for tambon_pcode, df_temporal_tambon in df_temporal.groupby(["tambon_pcode"]):
        # Max
        historical_max = df_temporal_tambon[columns].max(axis=1)
        df_temporal_tambon = df_temporal_tambon.assign(**{"max" : historical_max})
        df_temporal_tambon = df_temporal_tambon.assign(**{"pctl_max" : 100*historical_max.rank(pct=True)})
        
        # Min
        historical_min = df_temporal_tambon[columns].min(axis=1)
        df_temporal_tambon = df_temporal_tambon.assign(**{"min" : historical_min})
        df_temporal_tambon = df_temporal_tambon.assign(**{"pctl_min" : 100*historical_min.rank(pct=True)})

        # Med 
        historical_med = df_temporal_tambon[columns].median(axis=1)
        df_temporal_tambon = df_temporal_tambon.assign(**{"med" : historical_med})
        df_temporal_tambon = df_temporal_tambon.assign(**{"pctl_med" : 100*historical_med.rank(pct=True)})

        # Mean
        historical_mean = df_temporal_tambon[columns].mean(axis=1)
        df_temporal_tambon = df_temporal_tambon.assign(**{"mean" : historical_mean})
        df_temporal_tambon = df_temporal_tambon.assign(**{"pctl_mean" : 100*historical_mean.rank(pct=True)})
        
        list_df.append(df_temporal_tambon)
    df = pd.concat(list_df, ignore_index=True)
    return df

def extract_distribution_features(df_temporal, columns):
    df_temporal = df_temporal.copy()
    columns_no_period = [
        'no_period_in_hist_p0-p10' , 'no_period_in_hist_p10-p20', 'no_period_in_hist_p20-p30',
        'no_period_in_hist_p30-p40', 'no_period_in_hist_p40-p50', 'no_period_in_hist_p50-p60',
        'no_period_in_hist_p60-p70', 'no_period_in_hist_p70-p80', 'no_period_in_hist_p80-p90',
        'no_period_in_hist_p90-p100'
    ]
    columns_pct_period= [
        'pct_period_in_hist_p0-p10' , 'pct_period_in_hist_p10-p20', 'pct_period_in_hist_p20-p30',
        'pct_period_in_hist_p30-p40', 'pct_period_in_hist_p40-p50', 'pct_period_in_hist_p50-p60',
        'pct_period_in_hist_p60-p70', 'pct_period_in_hist_p70-p80', 'pct_period_in_hist_p80-p90',
        'pct_period_in_hist_p90-p100'
    ]
    
    list_df = []
    for tambon_pcode, df_temporal_tambon in df_temporal.groupby(["tambon_pcode"]):
        # Too slow
        # historical = df_temporal_tambon[columns].values.reshape(-1)
        # historical = historical[~np.isnan(historical)]
        # pctl_temporal = df_temporal_tambon[columns].applymap(lambda val: percentileofscore(historical, val), na_action='ignore')
        
        # Faster (50x)
        pctl_temporal = 100*df_temporal_tambon[columns].T.melt()["value"].rank(pct=True).values.reshape(df_temporal_tambon[columns].shape)
        df_temporal_tambon = df_temporal_tambon.assign(**{
            "no_period_in_hist_p0-p10"  :((pctl_temporal >= 0) & (pctl_temporal <= 10)).sum(axis=1),
            "no_period_in_hist_p10-p20" :((pctl_temporal > 10) & (pctl_temporal <= 20)).sum(axis=1),
            "no_period_in_hist_p20-p30" :((pctl_temporal > 20) & (pctl_temporal <= 30)).sum(axis=1),
            "no_period_in_hist_p30-p40" :((pctl_temporal > 30) & (pctl_temporal <= 40)).sum(axis=1),
            "no_period_in_hist_p40-p50" :((pctl_temporal > 40) & (pctl_temporal <= 50)).sum(axis=1),
            "no_period_in_hist_p50-p60" :((pctl_temporal > 50) & (pctl_temporal <= 60)).sum(axis=1),
            "no_period_in_hist_p60-p70" :((pctl_temporal > 60) & (pctl_temporal <= 70)).sum(axis=1),
            "no_period_in_hist_p70-p80" :((pctl_temporal > 70) & (pctl_temporal <= 80)).sum(axis=1),
            "no_period_in_hist_p80-p90" :((pctl_temporal > 80) & (pctl_temporal <= 90)).sum(axis=1),
            "no_period_in_hist_p90-p100":(pctl_temporal > 90).sum(axis=1)
        })
        df_temporal_tambon = df_temporal_tambon.assign(**{"total_period" : (~df_temporal_tambon[columns].isna()).sum(axis=1)})
        df_temporal_tambon[columns_pct_period] = 100*df_temporal_tambon[columns_no_period].divide(df_temporal_tambon["total_period"], axis=0)
        list_df.append(df_temporal_tambon)
    df = pd.concat(list_df, ignore_index=True)
    return df

def extract_intensity_features(df_temporal, columns):
    df_temporal = df_temporal.copy()
    # Intensity           
    list_df = []
    for tambon_pcode, df_temporal_tambon in df_temporal.groupby(["tambon_pcode"]):
        pctl_temporal = 100*df_temporal_tambon[columns].T.melt()["value"].rank(pct=True).values.reshape(df_temporal_tambon[columns].shape)
        # Under
        for pctl in [5, 10, 15, 20]:
            arr = (pctl_temporal < pctl).astype("int") 
            # Strict
            list_consecutive_strict = []
            for i in range(arr.shape[0]):
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_strict.append(0)
                else:
                    list_consecutive_strict.append(consecutive_one(arr[i]))
            # Relax
            list_consecutive_relax = []
            for i in range(arr.shape[0]):
                # Change from [1, 0 ,1] to [1, 1, 1] (window size = 3)
                for j in range(1, arr.shape[1]-1):
                    if (arr[i, j-1] == 1) and (arr[i, j+1] == 1):
                        arr[i, j] = 1
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_relax.append(0)
                else:
                    list_consecutive_relax.append(consecutive_one(arr[i]))
        
            df_temporal_tambon = df_temporal_tambon.assign(**{
                f"cnsct_period_under_{pctl}_strict" : list_consecutive_strict,
                f"cnsct_period_under_{pctl}_relax" : list_consecutive_relax
            })
            
        # Above
        for pctl in [80, 85, 90, 95]:
            arr = (pctl_temporal > pctl).astype("int")
            # Strict
            list_consecutive_strict = []
            for i in range(arr.shape[0]):
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_strict.append(0)
                else:
                    list_consecutive_strict.append(consecutive_one(arr[i]))
            # Relax
            list_consecutive_relax = []
            for i in range(arr.shape[0]):
                # Change from [1, 0 ,1] to [1, 1, 1] (window size = 3)
                for j in range(1, arr.shape[1]-1):
                    if (arr[i, j-1] == 1) and (arr[i, j+1] == 1):
                        arr[i, j] = 1
                # Find max consecutive
                if arr[i].sum() == 0:
                    list_consecutive_relax.append(0)
                else:
                    list_consecutive_relax.append(consecutive_one(arr[i]))
        
            df_temporal_tambon = df_temporal_tambon.assign(**{
                f"cnsct_period_above_{pctl}_strict" : list_consecutive_strict,
                f"cnsct_period_above_{pctl}_relax" : list_consecutive_relax
            })
        list_df.append(df_temporal_tambon)
    df = pd.concat(list_df, ignore_index=True)
    return df 
#%%
root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4`rootzone_features"
root_df_temporal =  r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\SMAP_L4_smrootzone_temporal"
list_p = np.unique([file.split(".")[0][-7:-5] for file in os.listdir(root_df_temporal)]).tolist()
#%%
columns_all = [f"t{i}" for i in range(181)]
#%%
# Loop for each province -> then each tambon
for p in list_p:
    print(p)
    # if p == "30":
    #     break
    df_temporal = pd.concat([pd.read_parquet(os.path.join(root_df_temporal, file)) for file in os.listdir(root_df_temporal) if file.split(".")[0][-7:-5] == p], ignore_index=True)
    df_temporal["tambon_pcode"] = "TH"+df_temporal["PLANT_PROVINCE_CODE"].astype(str).str.zfill(2)+df_temporal["PLANT_AMPHUR_CODE"].astype(str).str.zfill(2)+df_temporal["PLANT_TAMBON_CODE"].astype(str).str.zfill(2)
    # Extreme
    df_temporal = extract_extreme_features(df_temporal, columns_all)
    
    # Distribution
    df_temporal = extract_distribution_features(df_temporal, columns_all)
    
    # Intensity
    df_temporal = extract_intensity_features(df_temporal, columns_all)
    
    for year, df_temporal_grp in df_temporal.groupby("final_plant_year"):
        if len(df_temporal_grp) == 0:
            continue
        path_save = os.path.join(root_save, f"sm_features_{p}_{year}.parquet")
        df_temporal_grp.to_parquet(path_save)