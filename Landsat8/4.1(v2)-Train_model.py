import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import icedumpy
import warnings
warnings.filterwarnings('ignore')


def plot_roc_curve(model, x, y, label, color='b-'):
    y_predict_prob = model.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predict_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color, label=f"{label} (AUC = {auc:.4f})")
    plt.plot(fpr, fpr, "--")
    plt.xlabel("False Alarm")
    plt.ylabel("Detection")

    return y_predict_prob, fpr, tpr, thresholds, auc

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def get_file_dataframe(root, pathrows, bands):
    df = pd.DataFrame(columns=['p', 'pathrow', 'band', 'path_file'])
    for file in os.listdir(root):
        p = file.split("_")[4]
        pathrow = file.split("_")[5]
        band = file.split(".")[0].split("_")[6]

        if (pathrow in pathrows) and (band in bands):
            df = df.append(pd.Series({'p' : p,
                                      'pathrow' : pathrow,
                                      'band' : band,
                                      'path_file' : os.path.join(root, file)}),
                           ignore_index=True)
    return df

def merge_and_concat_file_dataframe(df_file, ratio=1):
    list_df = []
    for (p, pathrow), df_file_grp in tqdm(df_file.groupby(['p', 'pathrow'])):
        for index, row in df_file_grp.iterrows():
            df_temp = pd.read_parquet(row['path_file'])
            if len(df_temp)==0:
                continue
            
            band = row['band']
            df_temp.columns = df_temp.columns[:-4].tolist() + [f"{band}({item})" for item in df_temp.columns[-4:]]
            
            if not "df_merge_band" in locals():
                df_merge_band = df_temp
            else:
                df_merge_band = pd.merge(df_merge_band, df_temp, how='inner', on=df_temp.columns[:10].tolist())
        
        # After finishing load and merge all of the selected bands, drop duplicates and append df_merge_band to list_df
        if "df_merge_band" in locals():
            bands = pd.unique(df_file['band'])
            if df_merge_band.shape[1]==(10+len(bands)*4):
                # equivalent to subset = df_merge_band.columns[10:] but a lot faster
                df_merge_band = df_merge_band.drop_duplicates(subset=['row', 'col', 'START_DATE'])
                # Sampling down a bit :( (Needed, especially for no flood)
                if ratio!=1:
                    df_merge_band = df_merge_band.sample(frac=ratio)
                list_df.append(df_merge_band)
                del df_merge_band
                
    df = pd.concat(list_df, ignore_index=True)
    return df

def add_cloudmask(root_cloudmask, df):
    df = icedumpy.df_tools.set_index_for_loc(df, index=100000*df['row'] + df['col'])
    df["C(t-2)"] = 0
    df["C(t-1)"] = 0
    df["C(t+0)"] = 0
    df["C(t+1)"] = 0
    
    for pathrow, df_pathrow in tqdm(df.groupby(["pathrow"])):
        df_cloudmask = icedumpy.df_tools.load_ls8_cloudmask_dataframe(root_cloudmask, pathrow, filter_row_col=df.index)
        df_cloudmask = icedumpy.df_tools.set_index_for_loc(df_cloudmask, index=100000*df_cloudmask['row'] + df_cloudmask['col'])
        
        # Create column date with index
        dates_cloudmask = pd.Series([datetime.datetime.strptime(date.split("_")[-1], "%Y%m%d") for date in df_cloudmask.columns[3:]])
        dates_cloudmask.index += 3
        
        for start_date, df_pathrow_start_date in df_pathrow.groupby(['START_DATE']):
            start_date_column_index = dates_cloudmask[start_date<=dates_cloudmask].index[0] 
            
            try:
                df_cloudmask_selected = df_cloudmask.loc[df_pathrow_start_date.index, ['row', 'col', 'scene_id'] + df_cloudmask.columns[start_date_column_index-2:start_date_column_index+2].tolist()].fillna(0)
                df_cloudmask_selected.columns = ['row', 'col', 'pathrow'] + ["C(t-2)", "C(t-1)", "C(t+0)", "C(t+1)"]
                
#                assert (df.loc[(df["START_DATE"]==start_date) & (df["pathrow"]==pathrow)].index==df_cloudmask_selected.index).all()
                df.loc[(df["START_DATE"]==start_date) & (df["pathrow"]==pathrow), ["C(t-2)", "C(t-1)", "C(t+0)", "C(t+1)"]] = df_cloudmask_selected[["C(t-2)", "C(t-1)", "C(t+0)", "C(t+1)"]]
            except:
                continue
    return df
#%%
root_cloudmask = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_cloudmask"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_flood_value"
root_df_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_noflood_value"
#%%
pathrows = ['128048', '128049', '127048', '127049']
#pathrows = ['129048', '129049', '130048', '130049']

bands = ['B2', 'B3', 'B4', 'B5']
ratio_for_noflood = 0.1

cloudmask = True
#%%
# Get dataframe of file path of selecteh pathrow and band
df_file_flood = get_file_dataframe(root_df_flood, pathrows, bands)
df_file_noflood = get_file_dataframe(root_df_noflood, pathrows, bands)

# Load data from dataframe of file path
df_noflood = merge_and_concat_file_dataframe(df_file_noflood, ratio=ratio_for_noflood)
df_flood = merge_and_concat_file_dataframe(df_file_flood)

# Use only high loss ratio
df_flood = df_flood[(df_flood['loss_ratio'] >= 0.8) & (df_flood['loss_ratio'] <= 1)]

# Post-process after dataframes are loaded
columns_data = df_flood.columns[10:]
df_flood = df_flood[~(df_flood[columns_data]==0).any(axis=1)]
df_noflood = df_noflood[~(df_noflood[columns_data]==0).any(axis=1)]

print("The length of no flood dataframe is:", len(df_noflood))
print("The length of flood dataframe is:", len(df_flood))

print(df_noflood.groupby(['pathrow']).size())
print(df_flood.groupby(['pathrow']).size())
#%%
# Add cloudmask
if cloudmask:
    df_flood = add_cloudmask(root_cloudmask, df_flood)
    
    df_noflood = df_noflood.sample(n=min(2*len(df_flood), len(df_noflood)//3))
    df_noflood = add_cloudmask(root_cloudmask, df_noflood)
    
    # Remove too many clouds
    # Before removing cloud
    print("Before removing cloud")
    print("Flood(mean):    ", df_flood.iloc[:, -8:-4].mean().values)
    print("Non-Flood(mean):", df_noflood.iloc[:, -8:-4].mean().values)
    print()
    print("Flood(var):     ", df_flood.iloc[:, -8:-4].var().values)
    print("Non-Flood(var): ", df_noflood.iloc[:, -8:-4].var().values)
    
    print("-------------------------------------------------")
    df_flood = df_flood[(df_flood.iloc[:, -4:]==0).all(axis=1)]
    df_noflood = df_noflood[(df_noflood.iloc[:, -4:]==0).all(axis=1)]
    
    # After removing cloud
    print("After removing cloud")
    print("Flood(mean):    ", df_flood.iloc[:, -8:-4].mean().values)
    print("Non-Flood(mean):", df_noflood.iloc[:, -8:-4].mean().values)
    print()
    print("Flood(var):     ", df_flood.iloc[:, -8:-4].var().values)
    print("Non-Flood(var): ", df_noflood.iloc[:, -8:-4].var().values)
#%%
# Get train dev samples
#sample_size = min(100000, len(df_flood))
sample_size = min(len(df_flood), len(df_noflood))

samples_flood = df_flood.sample(n=sample_size)
samples_noflood = df_noflood.sample(n=sample_size)

x = pd.concat((samples_flood, samples_noflood))
x = x.assign(label=sample_size*[1] + sample_size*[0])
x = x.sample(frac=1)

x_train = x.iloc[:int(0.8*len(x))][columns_data].values
x_test = x.iloc[int(0.8*len(x)):][columns_data].values
y_train = x.iloc[:int(0.8*len(x)), -1].values
y_test = x.iloc[int(0.8*len(x)):, -1].values

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape, "Class count:", np.unique(y_train, return_counts=True)[1])
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape, "Class count:", np.unique(y_test, return_counts=True)[1])
#%%
# Define model and train
model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, random_state=0, verbose=10, n_jobs=-1)
model.fit(x_train, y_train)
score_train = model.score(x_train, y_train)
score_test = model.score(x_test, y_test)

print("score train:", score_train)
print("score_test:", score_test)
model.verbose = 0

# Plot ROC Curve of train and test
plt.close('all')
plt.figure(figsize=(16.0, 9.0))
y_predict_prob_train, fpr_train, tpr_train, thresholds_train, auc_train = plot_roc_curve(model, x_train, y_train, label='train', color='r-')
y_predict_prob_test, fpr_test, tpr_test, thresholds_test, auc_test = plot_roc_curve(model, x_test, y_test, label='test', color='b-')
plt.xticks(np.arange(-0.05, 1.05, 0.05))
plt.yticks(np.arange(0, 1., 0.1))
plt.xlim(left=-0.05, right=1.05)
plt.ylim(bottom=-0.05, top=1.05)
plt.grid()
plt.legend(loc = 4)
#%% Get miss and false alarm
## Get threshold with selected fpr
selected_fpr = 0.3
selected_thrershold = get_threshold_of_selected_fpr(fpr_test, thresholds_test, selected_fpr)

# Get y_predict
y_predict_test  = (y_predict_prob_test[:, 1]>=selected_thrershold)

TP_index = np.logical_and(y_test.astype('bool'), y_predict_test) # Only y=1 and y_pred=1
TN_index = np.logical_and(~y_test.astype('bool'), ~y_predict_test) # Only y=0 and y_pred=0
FN_index = np.logical_and(y_test.astype('bool'), ~y_predict_test) # Only y=1 and y_pred=0
FP_index = np.logical_and(~y_test.astype('bool'), y_predict_test)# Only y=0 and y_pred=1

df_TP = x.iloc[int(0.8*len(x)):].iloc[np.nonzero(TP_index)]
df_TN = x.iloc[int(0.8*len(x)):].iloc[np.nonzero(TN_index)]

df_FN = x.iloc[int(0.8*len(x)):].iloc[np.nonzero(FN_index)]
df_FP = x.iloc[int(0.8*len(x)):].iloc[np.nonzero(FP_index)]
#%%
root_save = os.path.join(r"F:\CROP-PIER\CROP-WORK\Model_visualization", f"model_{'_'.join(pathrows)}_{'_'.join(bands)}_cloud_0")
os.makedirs(root_save, exist_ok=True)
plt.savefig(os.path.join(root_save, "ROC.png"), transparent=True, dpi=200)
df_TP.to_parquet(os.path.join(root_save, "df_TP.parquet"))
df_TN.to_parquet(os.path.join(root_save, "df_TN.parquet"))
df_FN.to_parquet(os.path.join(root_save, "df_FN.parquet"))
df_FP.to_parquet(os.path.join(root_save, "df_FP.parquet"))
x.to_parquet(os.path.join(root_save, "x.parquet"))

icedumpy.io_tools.save_model(os.path.join(root_save, "model.joblib"), model)
icedumpy.io_tools.save_h5(path=os.path.join(root_save, "ROC_params.h5"), dict_save={'y_predict_prob_train':y_predict_prob_train, 'y_predict_prob_test':y_predict_prob_test,
                                                                                    'fpr_train':fpr_train,'fpr_test':fpr_test,
                                                                                    'tpr_train':tpr_train,'tpr_test':tpr_test,
                                                                                    'thresholds_train':thresholds_train,'thresholds_test':thresholds_test,
                                                                                    'auc_train':auc_train, 'auc_test':auc_test
                                                                                    })
#%% A
    

    
    
    
    
    
    
    
    
    
    
    