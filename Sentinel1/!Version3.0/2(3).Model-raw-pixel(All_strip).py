import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe, load_mapping
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.io_tools import save_model, save_h5
#%% Define functions
def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]
    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df
#%%
sat_type = "S1AB"
root_df_sentinel1 = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v4(at-False)"
#%%
root_df_flood = os.path.join(root_df_sentinel1, f"{sat_type.lower()}_flood_pixel")
root_df_nonflood = os.path.join(root_df_sentinel1, f"{sat_type.lower()}_nonflood_pixel")
#%%
list_df = []
for strip_id in ["302", "303", "304", "305", "306", "401", "402", "403"]:
    print(strip_id)
    # Load flood and non-flood dataframe
    df_flood = load_s1_flood_nonflood_dataframe(root_df_flood, p=None, strip_id=strip_id)
    df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p=None, strip_id=strip_id)

    # Replace negative data with nan
    df_flood = replace_zero_with_nan(df_flood)
    df_nonflood = replace_zero_with_nan(df_nonflood)

    # Drop row with nan (any)
    columns_pixel_values = df_flood.columns[df_flood.columns.str.contains(r'^t\d+$')].tolist()
    df_flood = df_flood.dropna(subset=columns_pixel_values)
    df_nonflood = df_nonflood.dropna(subset=columns_pixel_values)
    
    # Concat flood and non-flood and drop All_touched == True (Use All_touched=False)
    df_sample = pd.concat([df_flood, df_nonflood])
    df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id, list_p=list(map(str, df_sample["PLANT_PROVINCE_CODE"].unique())))
    df_sample = pd.merge(df_sample, df_mapping, how='inner', on=["new_polygon_id", "row", "col"], left_index=True)

    # Create label
    df_sample = df_sample.assign(label=(df_sample["loss_ratio"] != 0).astype("uint8"))
    
    # Drop duplicates
    df_sample = df_sample.drop_duplicates(subset=columns_pixel_values)
    
    # Drop Tier 2 polygon
    df_sample = df_sample[df_sample["tier"].isin([1])]
    
    # Select on loss_ratio = 0, between(0.8, 1.0)
    df_sample = df_sample.loc[(df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"].between(0.8, 1.0))]
    
    df_sample = df_sample.loc[  
        (df_sample["ext_act_id"].isin(np.random.choice(df_sample.loc[df_sample["label"] == 0, "ext_act_id"].unique(), 2*len(df_sample.loc[df_sample["label"] == 1, "ext_act_id"].unique()), replace=False)))
        | (df_sample["label"] == 1)
    ]

    list_df.append(df_sample)

df_sample = pd.concat(list_df, ignore_index=True)
# Create train-test samples
df_train = df_sample.loc[(~(df_sample["ext_act_id"]%10).isin([8, 9])) & ((df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"] >= 0.8))]
df_test = df_sample.loc[((df_sample["ext_act_id"]%10).isin([8, 9])) & ((df_sample["loss_ratio"] == 0) | (df_sample["loss_ratio"] >= 0.8))]
#%%
columns_pixel_values = df_train.columns[df_train.columns.str.contains(r'^t\d+$')].tolist()
model = RandomForestClassifier(n_estimators=100,
                               min_samples_leaf=5, max_depth=10, min_samples_split=10,
                               verbose=10, class_weight="balanced",
                               n_jobs=-1, random_state=42)
model.fit(df_train[columns_pixel_values].values, df_train["label"].values)
#%%
ax, _, _, _, _, _ = plot_roc_curve(model, df_train[columns_pixel_values], df_train["label"], "Train")
_, _, _, _, _, _ = plot_roc_curve(model, df_test[columns_pixel_values], df_test["label"], "Test", color="r--", ax=ax)
set_roc_plot_template(ax)
#%%
path_model = os.path.join(r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB", "all_RF_raw_pixel_values.joblib")
save_model(path_model, model)
y_predict_prob = model.predict_proba(df_train[columns_pixel_values].values)
fpr, tpr, threshold = metrics.roc_curve(df_train["label"].values, y_predict_prob[:, 1])
dict_roc_params = {"fpr":fpr,
                   "tpr":tpr,
                   "threshold":threshold}
save_h5(os.path.join(os.path.dirname(path_model), "all_RF_raw_pixel_values_roc_params.h5"),
        dict_roc_params)
#%%
from icedumpy.io_tools import load_model, load_h5
from tqdm import tqdm
def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%%
path_model = os.path.join(r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB", "all_RF_raw_pixel_values.joblib")
model = load_model(path_model)
path_roc_params = os.path.join(os.path.dirname(path_model), "all_RF_raw_pixel_values_roc_params.h5")
dict_roc_params = load_h5(path_roc_params)
model.verbose = 0
#%%
df_test = df_test.assign(pred_prob = model.predict_proba(df_test[columns_pixel_values])[:, 1])
df_test = df_test.assign(pred = (df_test["pred_prob"] >= 0.5).astype("uint8"))
#%%
metrics.confusion_matrix(df_test["label"], df_test["pred"])
#%%
dict_result = {"threshold":[], "precision":[], 
               "recall":[], "FPR":[], "NPV":[]}
for threshold in tqdm(np.arange(0, 1, 0.005)):
    df_test = df_test.assign(pred = (df_test["pred_prob"] >= threshold).astype("uint8"))
    cnf_matrix = metrics.confusion_matrix(df_test["label"], df_test["pred"])
    
    precision = cnf_matrix[1, 1]/(cnf_matrix[0, 1]+cnf_matrix[1, 1])
    recall = cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])
    FPR = cnf_matrix[0, 1]/(cnf_matrix[0, 1]+cnf_matrix[0, 0])
    NPV = cnf_matrix[0, 0]/(cnf_matrix[0, 0]+cnf_matrix[1, 0])
    
    dict_result["threshold"].append(threshold)
    dict_result["precision"].append(precision)
    dict_result["recall"].append(recall)
    dict_result["FPR"].append(FPR)
    dict_result["NPV"].append(NPV)
    # print(f"{threshold:.3f}, {precision:.3f}, {recall:.3f}, {FPR:.3f}")
df_result = pd.DataFrame(dict_result)
#%%