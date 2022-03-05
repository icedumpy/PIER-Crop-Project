import os
import rasterio
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
#%%
dict_classes = {
    "green": [5, 6, 12],
    "water": [2, 11, 19],
    "mixed": [16, 17, 18]
}
#%%
raster_name = "petchabun_output"
# Read shapefile
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\shapefile\gdf_sel_plot_ks_plant_info.shp")
# gdf = gdf[gdf["t_code"] == 460305]
# Read raster
raster = rasterio.open(rf"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\{raster_name}.tif")
root_save_shp = rf"F:\CROP-PIER\CROP-WORK\FW__Update_on_Drone_image_analysis\{raster_name}"
os.makedirs(root_save_shp, exist_ok=True)
#%%
for index, series in tqdm(gdf.iterrows(), total=len(gdf)):
    polygon = series.geometry
    try:
        arr, _ = mask(
            raster, 
            [polygon], 
            crop=True, 
            all_touched=False,
            indexes=1,
            nodata=-1
        )
    except ValueError:
        continue
    # Count unique of each cluster
    arr = arr[arr != -1]
    unique, counts = np.unique(arr, return_counts=True)
    dict_unique_count = dict(zip(unique, counts/len(arr)))
    
    # Create training dataset
    for key in dict_classes.keys():
        # Count portion of each class
        class_portion = sum([dict_unique_count[cluster_num] if cluster_num in dict_unique_count.keys() else 0 for cluster_num in dict_classes[key]])
        gdf.loc[index, key] = class_portion
    # This is label >> "loss_f_adj"
gdf = gdf[~gdf[['green', 'water', 'mixed']].isna().any(axis=1)]
#%% Model: Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#%%
x = gdf[["green", "water", "mixed"]]
y = gdf["loss_f_adj"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=1000, max_depth=5, n_jobs=-1)
model.fit(x_train, y_train)
print(f"Train Accuracy: {model.score(x_train, y_train):.2f}")
print(f"Test  Accuracy: {model.score(x_test , y_test ):.2f}")    
#%%
# # Parameters tuning
rf = RandomForestClassifier(n_jobs=-1)
random_grid = {
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5, None],
    'max_features': ['auto'],
    'min_samples_leaf': [2, 5],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 200]
}
rf_grid = GridSearchCV(estimator=rf, param_grid=random_grid, scoring="f1_weighted", cv=3, verbose=2, n_jobs=4)
# Fit the random search model
rf_grid.fit(x_train, y_train)
print(rf_grid.best_params_)
#%%
cnf_matrix = confusion_matrix(y_test, rf_grid.predict(x_test))    
disp = plot_confusion_matrix(rf_grid, x_test, y_test)
rf_grid.score(x_test, y_test)
plt.savefig(os.path.join(root_save_shp, "cnf_matrix.png"), bbox_inches="tight")
#%%
import pandas as pd
df_test = pd.concat([x_test, y_test], axis=1)
df_test["y_pred"] = rf_grid.predict(x_test)
df_test = df_test.join(gdf["geometry"])
df_test = gpd.GeoDataFrame(df_test)
df_test["Pred Act"] = df_test["loss_f_adj"].astype(str)+" "+df_test["y_pred"].astype(str)
df_test.to_file(os.path.join(root_save_shp, "pred_test.shp"))
#%%
gdf["y_argmax"] = np.argmax(gdf[["green", "mixed","water",]].values, axis=1)
gdf["Act Pred"] = gdf["y_argmax"].astype(str)+" "+gdf["loss_f_adj"].astype(str)
gdf.to_file(os.path.join(root_save_shp, "argmax.shp"))
#%%
cnf_matrix = confusion_matrix(gdf["loss_f_adj"], gdf["y_argmax"])
from sklearn.metrics import accuracy_score
accuracy_score(gdf["loss_f_adj"], gdf["y_argmax"])


















