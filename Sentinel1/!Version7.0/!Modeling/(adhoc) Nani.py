import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#%%
def fit_and_eval(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    f1 = f1_score(y_test, model.predict(x_test))
    print(f"F1({model}): {f1:.4f}")
    return f1

def get_list_model():
    list_model = [
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=1, max_iter=1000),
    ]
    return list_model
#%% Load train test data
df_manually_train = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed_mannually_selected_train.parquet")
df_manually_test  = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed_mannually_selected_test.parquet")
df_rfe_train = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed_rfe_train.parquet")
df_rfe_test  = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed_rfe_test.parquet")
#%%
dict_f1 = dict()

# Manually
x_train = df_manually_train.iloc[:, 1:]
y_train = df_manually_train["y"]
x_test = df_manually_test.iloc[:, 1:]
y_test = df_manually_test["y"]
print("Manually selected features")
dict_f1["Manually"] = dict()
for model in get_list_model():
    list_f1 = []
    for i in range(10):
        f1 = fit_and_eval(model, x_train, y_train, x_test, y_test)
        list_f1.append(f1)
    dict_f1["Manually"][str(model)] = np.array(list_f1).mean()

# RFE
x_train = df_rfe_train.iloc[:, 1:]
y_train = df_rfe_train["y"]
x_test = df_rfe_test.iloc[:, 1:]
y_test = df_rfe_test["y"]
print("RFE selected features")
dict_f1["RFE"] = dict()
for model in get_list_model():
    list_f1 = []
    for i in range(10):
        f1 = fit_and_eval(model, x_train, y_train, x_test, y_test)
        list_f1.append(f1)
    dict_f1["RFE"][str(model)] = np.array(list_f1).mean()
df_f1 = pd.DataFrame(dict_f1)
#%%





















