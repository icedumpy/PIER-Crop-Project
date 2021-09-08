from sklearn import metrics
import matplotlib.pyplot as plt
from icedumpy.io_tools import load_h5, load_model
#%%
model = load_model(r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\303\303_RF_raw_pixel_values.joblib")
params = load_h5(r"F:\CROP-PIER\CROP-WORK\Model\sentinel1\S1AB\303\303_RF_raw_pixel_values_roc_params.h5")
fpr = params["fpr"]
tpr = params["tpr"]
#%%
fig, ax = plt.subplots()
auc = metrics.auc(fpr, tpr)
ax.plot(fpr, tpr, label=f"Ice (AUC = {auc:.4f})")
ax.plot(fpr, fpr, "--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
#%%