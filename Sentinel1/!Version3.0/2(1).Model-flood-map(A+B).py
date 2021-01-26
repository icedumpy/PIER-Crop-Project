#%% Import libraries
import os
import datetime
import pandas as pd
import numpy as np
from numba import jit
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe
from icedumpy.geo_tools import create_tiff, gdal_descriptions, create_vrt
#%% Define functions
datetime_to_yyyymmdd = lambda date: "".join(str(date.date()).split("-"))

def create_trs(path_dst, path_src, bandList=None, file_format='ENVI'):
    outds = gdal.Translate(path_dst, path_src, bandList=bandList, format=file_format)
    outds.FlushCache()
    del outds

def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]

    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df

def add_change_columns(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r'^t\d+$')].tolist()
    columns_diff = [f"{columns_pixel_values[i]}-{columns_pixel_values[i+1]}" for i in range(len(columns_pixel_values)-1)]
    df = df.assign(**{column:values for column, values in zip(columns_diff, -np.diff(df.loc[:, columns_pixel_values]).T)})
    return df, columns_pixel_values, columns_diff

def get_thresholds(root_df_flood, p=None, strip_id=None,
                   percentile_water=0.8,
                   percentile_change_to_water=0.2,
                   percentile_change_from_water=0.8):
    # Load data
    df_flood = load_s1_flood_nonflood_dataframe(root_df=root_df_flood, p=p, strip_id=strip_id)
    
    # Train only 80% (ext_act_id in range(0, 8))
    df_flood = df_flood[~(df_flood["ext_act_id"]%10).isin([8, 9])]
    
    # Replace negative data with nan
    df_flood = replace_zero_with_nan(df_flood)
    
    # Select only high loss ratio (only flood)
    df_flood = df_flood[(df_flood['loss_ratio'] >= 0.8) & (df_flood['loss_ratio'] <= 1.0)]
    
    # Add change value columns (new-old)
    df_flood, columns_pixel_values, columns_diff = add_change_columns(df_flood)
    
    # Drop row with all nan
    df_flood = df_flood[(~df_flood[columns_pixel_values].isna().all(axis=1)) | (~df_flood[columns_diff].isna().all(axis=1))]

    # Get thresholds
    threshold_water = df_flood[columns_pixel_values].min(axis=1).quantile(percentile_water)
    threshold_change_to_water = df_flood[columns_diff].max(axis=1).quantile(percentile_change_to_water)
    threshold_change_from_water = df_flood[columns_diff].min(axis=1).quantile(percentile_change_from_water)   
    
    return threshold_water, threshold_change_to_water, threshold_change_from_water

def create_flood_map(root_raster_flood_map, path_raster, strip_id, repeat_cycle, threshold_water, threshold_change_to_water, threshold_change_from_water, folder_name):
    raster = gdal.Open(path_raster)
    projection = raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    df_raster_date = pd.DataFrame(pd.to_datetime([raster.GetRasterBand(band_number).GetDescription()[:8] for band_number in range(3, raster.RasterCount+1)]), columns=["raster_date"])
    df_raster_date.index+=3 # 1, 2 is row, col
    df_raster_date = df_raster_date.assign(diff_date_from_previous_row = (df_raster_date["raster_date"] - df_raster_date["raster_date"].shift()).dt.days)
    df_raster_date = df_raster_date.assign(sat_type = [raster.GetRasterBand(band_number).GetDescription()[-3:] for band_number in range(3, raster.RasterCount+1)])
    
    # Initialize flood map
    flood_map = np.zeros((raster.RasterYSize, raster.RasterXSize), dtype="bool")
    
    # Initialize dictionary for raster images
    dict_raster_images = dict()
    
    for idx, band_num in enumerate(df_raster_date.index):
        date = "".join(str(df_raster_date.at[band_num, "raster_date"].date()).split("-"))
        diff_date = df_raster_date.at[band_num, 'diff_date_from_previous_row']
        sat_type = df_raster_date.at[band_num, 'sat_type']
        
        path_raster_flood_map = os.path.join(root_raster_flood_map, strip_id, folder_name, f"{date}_{sat_type}.tif")
        if os.path.exists(path_raster_flood_map):
            continue
        os.makedirs(os.path.dirname(path_raster_flood_map), exist_ok=True)    
        # Start process (if diff_date is not NaN and diff_date == repeat_cycle)
        if ~pd.isna(diff_date) and diff_date == repeat_cycle:
            band_previous = band_num-1
            band_current  = band_num
            # Step1: Remove unused images and add new images into dictionary
            [dict_raster_images.pop(key) for key in [key for key in dict_raster_images.keys() if not key in [band_previous, band_current]]]
            if not band_previous in dict_raster_images.keys():
                dict_raster_images[band_previous] = raster.GetRasterBand(band_previous).ReadAsArray()
            if not band_current in dict_raster_images.keys():
                dict_raster_images[band_current] = raster.GetRasterBand(band_current).ReadAsArray()
            
            # Step2: Calculate diff_image (Previous-Current)
            diff_image = dict_raster_images[band_previous]-dict_raster_images[band_current]
            
            # Step3: Find water mask (current_raster_image <= threshold_water)
            mask_water = dict_raster_images[band_current] <= threshold_water
            
            # Step4: Find change_to_water mask (diff_image >= threshold_change_to_water)
            mask_change_to_water = diff_image >= threshold_change_to_water
            
            # Step5: Find change_from_water mask (diff_image <= threshold_change_from_water)
            mask_change_from_water = diff_image <= threshold_change_from_water
            
            # Step6: Step3 + Step4 + Step5 (Derived from AJ.Teerasit code)
            # 1. flood_map = (flood_map | to_water)
            # 2. flood_map = flood_map & (water & ~from_water)
            # 1+2. flood_map = (flood_map | to_water) & (water & ~from_water)
            flood_map = (flood_map | mask_change_to_water) & (mask_water & ~mask_change_from_water)
            
        # Save flood map (Save every loop (including skip image))
        create_tiff(path_save=path_raster_flood_map,
                    im=np.expand_dims(flood_map.astype("uint8"), axis=0),
                    projection=projection,
                    geotransform=geotransform,
                    drivername="GTiff",
                    nodata=0,
                    dtype="uint8"
                    )
    return os.path.dirname(path_raster_flood_map)

def create_flood_map_from_percentiles(root_df_flood, root_raster_flood_map, root_temp, path_raster, 
                                      p, strip_id, repeat_cycle, percentile_template, 
                                      percentile_water,
                                      percentile_change_to_water,
                                      percentile_change_from_water):
    # Get threshold from the selected percentiles
    threshold_water, threshold_change_to_water, threshold_change_from_water = get_thresholds(root_df_flood, 
                                                                                             p, 
                                                                                             strip_id,
                                                                                             percentile_water,
                                                                                             percentile_change_to_water,
                                                                                             percentile_change_from_water)
    # Create flood map
    folder_raster_flood_map = create_flood_map(root_raster_flood_map, path_raster, strip_id, repeat_cycle, threshold_water, threshold_change_to_water, threshold_change_from_water, percentile_template)
    return folder_raster_flood_map

def stack_flood_map(folder_raster_flood_map, root_temp):
    list_path_file = []
    list_file = []
    list_sat_type = []
    for file in os.listdir(folder_raster_flood_map):
        path_file = os.path.join(folder_raster_flood_map, file)
        list_path_file.append(path_file)
        list_file.append(file)
        list_sat_type.append(file.split(".")[0][-3:])
        
    df_file = pd.DataFrame({"path_file":list_path_file, "file":list_file,
                            "sat_type":list_sat_type}, columns=["path_file", "file", "sat_type"])
    df_file = df_file.sort_values(["file"])
    df_file = df_file.assign(band_name=df_file["file"].str.split(".").str[0])
    
    strip_id = os.path.basename(os.path.dirname(folder_raster_flood_map))
    list_path_raster = df_file["path_file"].tolist()
    list_band_name = df_file["band_name"].tolist()
    path_vrt = os.path.join(root_temp, "Stacked_flood_map", f"{strip_id}.vrt")
    os.makedirs(os.path.dirname(path_vrt), exist_ok=True)
    src_nodata = 0
    dst_nodata = 0
    create_vrt(path_vrt, list_path_raster, list_band_name, src_nodata, dst_nodata)
    return path_vrt

def load_raster_images_of_selected_bands(raster, list_bands, dict_img=dict()):
    [dict_img.pop(key) for key in [key for key in dict_img.keys() if not key in list_bands]]
    for band in list_bands:
        if not band in dict_img.keys():
            band = int(band)
            dict_img[band] = raster.GetRasterBand(band).ReadAsArray()
    return dict_img

def get_flood_map_pixel_values(path_raster_flood, df_flood_nonflood, repeat_cycle, period_before, period_after):
    # Repeat_cycle >> A: 12, B: 12, AB: 6
    raster_flood_map = gdal.Open(path_raster_flood)
    list_raster_flood_map_date = gdal_descriptions(raster_flood_map)
    
    before_raster_flood_first_date = datetime.datetime.strptime(list_raster_flood_map_date[0][:8], "%Y%m%d")-datetime.timedelta(2*repeat_cycle*period_before)
    after_raster_flood_first_date = datetime.datetime.strptime(list_raster_flood_map_date[-1][:8], "%Y%m%d")+datetime.timedelta(2*repeat_cycle*period_after)

    df_flood_raster_date = pd.DataFrame(pd.date_range(before_raster_flood_first_date, after_raster_flood_first_date, freq=f'{repeat_cycle}D'), columns=['ideal'])
    df_flood_raster_date.loc[:, "ideal"] = df_flood_raster_date.loc[:, "ideal"].astype(str).str.split("-").str.join("")
    df_flood_raster_date.loc[df_flood_raster_date.index%2 == 0, "sat_type"] = list_raster_flood_map_date[0].split("_")[1]
    df_flood_raster_date.loc[df_flood_raster_date.index%2 == 1, "sat_type"] = list(set(["S1A", "S1B"]) - set([list_raster_flood_map_date[0].split("_")[1]]))[0]
    df_flood_raster_date['available'] = df_flood_raster_date['ideal'].isin(list(map(lambda val: val[:8], list_raster_flood_map_date)))
    df_flood_raster_date['raster_band'] = (df_flood_raster_date['available'].cumsum()).astype('int')
    df_flood_raster_date.loc[~df_flood_raster_date['available'], 'raster_band'] = 0

    dict_img = dict()
    list_df = list()
    for anchor_date, df_flood_nonflood_grp in df_flood_nonflood.groupby(["ANCHOR_DATE"]):
        try:
            before_anchor_date = datetime_to_yyyymmdd(anchor_date-datetime.timedelta(period_before*repeat_cycle))
            after_anchor_date  = datetime_to_yyyymmdd(anchor_date+datetime.timedelta((period_after-1)*repeat_cycle))
            anchor_date = datetime_to_yyyymmdd(anchor_date)
            
            index_first = np.where(df_flood_raster_date["ideal"] >= before_anchor_date)[0][0]
            index_last = np.where(df_flood_raster_date["ideal"] >= after_anchor_date)[0][0]
            
            df_flood_raster_date_window = df_flood_raster_date.loc[index_first:index_last].reset_index(drop=True)
            
            list_bands = [band for band in df_flood_raster_date_window["raster_band"] if band != 0]
            
            dict_img = load_raster_images_of_selected_bands(raster_flood_map, list_bands, dict_img)
            flood_img = np.array(list(dict_img.values()))
            
            rows = df_flood_nonflood_grp["row"]
            cols = df_flood_nonflood_grp["col"]
            df_temp = pd.DataFrame(flood_img[:, rows, cols].T, columns=df_flood_raster_date_window.index[df_flood_raster_date_window['available']].astype(str).tolist())
            if (~df_flood_raster_date_window["available"]).any():
                df_temp = df_temp.assign(**{str(index):np.nan for index in df_flood_raster_date_window.index[~df_flood_raster_date_window["available"]]})
                df_temp = df_temp.sort_index(axis=1)
            # df_temp.columns = [(f"{item}_{i//2}") for i, item in enumerate(df_flood_raster_date_window["sat_type"], start=2)]
            df_temp = df_temp.assign(first_sat=df_flood_raster_date_window.at[0, "sat_type"])
            df_temp.index = df_flood_nonflood_grp.index
            list_df.append(df_temp)
        except:
            continue
    df = pd.concat(list_df)
    df = pd.concat([df_flood_nonflood, df], axis=1)
    # assert (df.index == df_flood_nonflood.index).all()
    return df

def get_df_flood_map(root_df_flood, root_df_nonflood, path_raster_flood, p, strip_id, repeat_cycle, period_before, period_after):
    
    df_flood = load_s1_flood_nonflood_dataframe(root_df_flood, p, strip_id=strip_id)
    df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p, strip_id=strip_id)
    
    # Replace negative data with nan
    df_flood = replace_zero_with_nan(df_flood)
    df_nonflood = replace_zero_with_nan(df_nonflood)
 
    # Drop row with all nan
    columns_pixel_values = df_flood.columns[df_flood.columns.str.contains(r'^t\d+$')].tolist()
    df_flood = df_flood.loc[~pd.isna(df_flood.loc[:, columns_pixel_values]).all(axis=1)]
    df_nonflood = df_nonflood.loc[~pd.isna(df_nonflood.loc[:, columns_pixel_values]).all(axis=1)]
    
    # Test data (only-nonflood)
    # df_flood = df_flood[(df_flood["ext_act_id"]%10).isin([8, 9])]
    
    # Test data (only high loss ratio)
    # df_flood = df_flood[(df_flood["loss_ratio"] >= 0.8)]
    
    # Sampling df_nonflood (equal ext_act_id as df_flood)
    df_nonflood_unique = df_nonflood["ext_act_id"].unique()
    df_nonflood = df_nonflood[df_nonflood["ext_act_id"].isin(df_nonflood_unique[np.random.permutation(len(df_nonflood_unique))[:len(df_flood["ext_act_id"].unique())]])]
    
    # Add flood map data
    df_flood = get_flood_map_pixel_values(path_raster_flood, df_flood, repeat_cycle, period_before, period_after)
    df_nonflood = get_flood_map_pixel_values(path_raster_flood, df_nonflood, repeat_cycle, period_before, period_after)
    
    return df_flood, df_nonflood

@jit(nopython=True)
def find_max_consecutive(arr):
    list_max_consecutive = []
    for i in range(arr.shape[0]):
        vector = arr[i]
        flag = False
        consecutive = 0
        max_consecutive = 0
        for idx, j in enumerate(vector):
            # if element == 1, consecutive+=1, Raise flag
            if j == 1:
                flag = True
                consecutive +=1
            # if element == 0 and flag == True, reset consecutive count
            if (j == 0 and flag == True):
                if consecutive > max_consecutive:
                    max_consecutive = consecutive
                flag = False
                consecutive = 0
        if consecutive > max_consecutive:
            max_consecutive = consecutive
        list_max_consecutive.append(max_consecutive)
    return list_max_consecutive

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]
#%% Define directories
root_temp = r"C:\Users\PongporC\Desktop\temp"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
root_df_nonflood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
#%% Define parameters
sat_type = "S1AB"
p = None
strip_id = "402"
repeat_cycle = 6
period_before = 4 
period_after = 4

# Add path
root_temp = os.path.join(root_temp, sat_type)
root_df_flood = os.path.join(root_df_flood, f"{sat_type.lower()}_flood_pixel")
root_df_nonflood = os.path.join(root_df_nonflood, f"{sat_type.lower()}_nonflood_pixel")
root_raster_flood_map = os.path.join(root_temp, "Flood_map")
#%% Execute#1 Create translated raster for flood map
path_vrt = os.path.join(root_temp, f"{sat_type}_{strip_id}.vrt") # Virtual path
path_trs = os.path.splitext(path_vrt)[0] # Translated path
#%%
if not os.path.exists(path_trs):
    raster = gdal.Open(path_vrt)
    create_trs(path_trs, path_vrt, bandList=[idx for idx, band in enumerate(gdal_descriptions(raster), start=1) if band[:4] != "2020"])
#%% Create flood map and stack
percentile_water, percentile_change_to_water = 0.90, 0.90
percentile_change_from_water = 0.7
        
percentile_change_to_water = 1-percentile_change_to_water
percentile_template = f"{percentile_water:.3f}_{percentile_change_to_water:.3f}_{percentile_change_from_water:.3f}"

folder_raster_flood_map = create_flood_map_from_percentiles(root_df_flood, root_raster_flood_map, root_temp, path_trs, 
                                                            p, strip_id, repeat_cycle, percentile_template, 
                                                            percentile_water,
                                                            percentile_change_to_water,
                                                            percentile_change_from_water)
path_raster_flood = stack_flood_map(folder_raster_flood_map, os.path.dirname(root_temp))
#%% Get flood map samples
df_flood, df_nonflood = get_df_flood_map(root_df_flood, root_df_nonflood, path_raster_flood, p=p, strip_id=strip_id,
                                         repeat_cycle=repeat_cycle, period_before=period_before, period_after=period_after)
columns_pixel_values = df_flood.columns[df_flood.columns.str.match(r"^t\d$")]
columns_flood_pixel_values = df_flood.columns[df_flood.columns.str.match(r"\d$")]
#%% Get train, test samples
df_sample = pd.concat([df_flood, df_nonflood])
df_sample = df_sample.assign(label=(df_sample["loss_ratio"] != 0).astype("uint8"))
df_sample = df_sample.sample(frac=1.0)
df_sample_dropdub = df_sample.drop_duplicates(subset=columns_pixel_values)

del df_flood, df_nonflood
#%% Re-create train test
df_train = df_sample_dropdub.loc[~(df_sample_dropdub["ext_act_id"]%10).isin([8, 9]) & ((df_sample_dropdub["loss_ratio"] == 0) | df_sample_dropdub["loss_ratio"] >= 0.8)]
df_test = df_sample_dropdub.loc[(df_sample_dropdub["ext_act_id"]%10).isin([8, 9]) & ((df_sample_dropdub["loss_ratio"] == 0) | df_sample_dropdub["loss_ratio"] >= 0.8)]

df_train = df_train.dropna(subset = columns_pixel_values)
df_test = df_test.dropna(subset = columns_pixel_values)
#%% Raw pixel values
model = RandomForestClassifier(n_jobs=-1, random_state=42, min_samples_leaf=5, max_depth=10, min_samples_split=10, verbose=0, class_weight="balanced")
model.fit(df_train[columns_pixel_values].values, df_train["label"].values)
#%%  Test
df_test = df_sample.loc[(df_sample["ext_act_id"]%10).isin([8, 9])]
df_test = df_test.dropna(subset = columns_pixel_values)
#%%
y_prob = model.predict_proba(df_test[columns_pixel_values])[:, 1]
df_test = df_test.assign(pred_prob = y_prob)
#%%
list_actual_loss_ratio = []
list_prob_loss_ratio = []
for (ext_act_id, loss_ratio), df_test_grp in tqdm(df_test.groupby(["ext_act_id", "loss_ratio"])):
    list_actual_loss_ratio.append(loss_ratio)
    list_prob_loss_ratio.append(df_test_grp["pred_prob"].mean())
#%%
temp = pd.DataFrame([list_actual_loss_ratio, list_prob_loss_ratio]).T
#%%
# plt.scatter(list_actual_loss_ratio, list_prob_loss_ratio)
plt.hist2d(list_actual_loss_ratio, list_prob_loss_ratio, bins=20)
plt.colorbar()
#%%
# # ROC Visualization
# fig, ax = plt.subplots(figsize=(16, 9))
# ax, _, fpr, _, thresholds, auc = plot_roc_curve(
#     ax, model, 
#     df_train[columns_pixel_values].values, 
#     df_train["label"].values,
#     "Train", color="g-"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model, 
#     df_test[columns_pixel_values].values, 
#     df_test["label"].values, 
#     "Test", color="b:"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model,
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2018), columns_pixel_values].values, 
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2018), "label"].values, 
#     "Test_2018", color="r--"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model,
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2019), columns_pixel_values].values, 
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2019), "label"].values, 
#     "Test_2019", color="y-."
# )
# ax = set_roc_plot_template(ax)
# ax.set_title(f"{strip_id}: (Flood, NonFlood)\nTrain: ({df_train['label'].values.sum()}, {(df_train['label'].values==0).sum()}),\nTest: ({df_test['label'].values.sum()}, {(df_test['label'].values==0).sum()}),\nTest_2018: ({df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2018), 'label'].values.sum()}, {(df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2018), 'label'].values==0).sum()}),\nTest_2019: ({df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2019), 'label'].values.sum()}, {(df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2019), 'label'].values==0).sum()})")
# fig.savefig(os.path.join(r"C:\Users\PongporC\Desktop\Adhoc", f"{strip_id}-raw-pixel-rfc.png"))
# #%% Flood map pixel value
# model = RandomForestClassifier(n_jobs=-1, random_state=42, min_samples_leaf=5, max_depth=10, min_samples_split=10, verbose=0, class_weight="balanced")
# model.fit(df_train[columns_flood_pixel_values].values, df_train["label"].values)

# # ROC Visualization
# fig, ax = plt.subplots(figsize=(16, 9))
# ax, _, fpr, _, thresholds, auc = plot_roc_curve(
#     ax, model, 
#     df_train[columns_flood_pixel_values].values, 
#     df_train["label"].values,
#     "Train", color="g-"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model, 
#     df_test[columns_flood_pixel_values].values, 
#     df_test["label"].values, 
#     "Test", color="b:"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model,
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2018), columns_flood_pixel_values].values, 
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2018), "label"].values, 
#     "Test_2018", color="r--"
# )
# ax, _, _, _, _, _ = plot_roc_curve(
#     ax, model,
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2019), columns_flood_pixel_values].values, 
#     df_test.loc[(df_test["ANCHOR_DATE"].dt.year == 2019), "label"].values, 
#     "Test_2019", color="y-."
# )
# ax = set_roc_plot_template(ax)
# ax.set_title(f"{strip_id}: (Flood, NonFlood)\nTrain: ({df_train['label'].values.sum()}, {(df_train['label'].values==0).sum()}),\nTest: ({df_test['label'].values.sum()}, {(df_test['label'].values==0).sum()}),\nTest_2018: ({df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2018), 'label'].values.sum()}, {(df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2018), 'label'].values==0).sum()}),\nTest_2019: ({df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2019), 'label'].values.sum()}, {(df_test.loc[(df_test['ANCHOR_DATE'].dt.year == 2019), 'label'].values==0).sum()})")
# fig.savefig(os.path.join(r"C:\Users\PongporC\Desktop\Adhoc", f"{strip_id}-floodmap-pixel-rfc.png"))
#%%