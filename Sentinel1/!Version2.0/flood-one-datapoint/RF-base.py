import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#%%
root_df_dry = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_noflood_pixel_from_mapping_v3"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1-dataframe-updated\s1_flood_pixel_from_mapping_v3"
#%%
index_name = 's304'
df_flood = pd.concat([pd.read_parquet(os.path.join(root_df_flood, file)) for file in os.listdir(root_df_flood) if index_name in file], ignore_index=True)
df_dry = pd.concat([pd.read_parquet(os.path.join(root_df_dry, file)) for file in os.listdir(root_df_dry) if index_name in file], ignore_index=True)
#%% Clean: นาปีี
df_flood = df_flood[((df_flood['final_plant_date'].dt.month>=5) & (df_flood['final_plant_date'].dt.month<=10))]
df_dry = df_dry[((df_dry['final_plant_date'].dt.month>=5) & (df_dry['final_plant_date'].dt.month<=10))]
#%% Clean 1: Drop nan
df_flood = df_flood[~np.any(df_flood[['t-2', 't-1', 't', 't+1']].isna(), axis=1)]
df_dry = df_dry[~np.any(df_dry[['t-2', 't-1', 't', 't+1']].isna(), axis=1)]
#%% Clean 2: Drop low loss raio
df_flood = df_flood[(df_flood['loss_ratio']>=0.8) & (df_flood['loss_ratio']<=1)]
#%% Clean 3: Drop dup
df_flood = df_flood.drop_duplicates(subset=['t-2', 't-1', 't', 't+1'])
df_dry = df_dry.drop_duplicates(subset=['t-2', 't-1', 't', 't+1'])
#%% Get sample
sample_size = len(df_flood)
#sample_size = 300000
samples_flood = df_flood[['t-2', 't-1', 't', 't+1']].sample(n=sample_size).values
samples_dry = df_dry[['t-2', 't-1', 't', 't+1']].sample(n=sample_size).values
num_flood_pixels = samples_flood.shape[0]
#%%
X = np.concatenate((samples_flood, samples_dry), axis=0)
Y = np.zeros((num_flood_pixels*2))
Y[:num_flood_pixels] = 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#%%
clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, random_state=0, verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
#%%
Y_predict = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, Y_predict)
print(conf_matrix)

diag = np.diag(conf_matrix)
precision = diag[1]/Y_predict.sum()
recall = diag[1]/y_test.sum() 
F1 = 2.0*(precision * recall) / (precision + recall)
acc = diag.sum()/ conf_matrix.sum()
print(f"Precision: {precision}")    
print(f"Recall: {recall}")
print(f"F1: {F1}")
print(f"Acc: {acc*100} %")
#%%
Y_predict_prob = clf.predict_log_proba(X_test)
log_likelihood = Y_predict_prob[:,1]- Y_predict_prob[:,0]
print(log_likelihood.max(), log_likelihood.min())

fpr, tpr, thresholds = metrics.roc_curve(y_test, log_likelihood)
plt.figure()
plt.plot(fpr, tpr, "b-", fpr, fpr, "--")
plt.xlabel("False Alarm")
plt.ylabel("Detection")
plt.grid()
#%%
ice = clf.predict_log_proba(X_train)
log_likelihood = ice[:,1]- ice[:,0]
print(log_likelihood.max(), log_likelihood.min())

fpr, tpr, thresholds = metrics.roc_curve(y_train, log_likelihood)
plt.plot(fpr, tpr, "r-", fpr, fpr, "--")