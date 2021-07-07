import os
import numpy as np
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_h5
def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%%
dict_roc_params = load_h5(r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\Model-season-v3(for-comparison)\402_metrics_params_train_2018_2019_2020.h5")
#%%
tpr = dict_roc_params["tpr_train"]
fpr = dict_roc_params["fpr_train"]

plt.plot(fpr, tpr)
plt.grid()
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
