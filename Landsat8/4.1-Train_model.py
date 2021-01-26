import os
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import icedumpy
import itertools

def plot_roc_curve(model, x, y, label, color='b-'):
    y_predict_prob = model.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predict_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color, label=f"{label} (AUC = {auc:.4f})")
    plt.plot(fpr, fpr, "--")
    plt.xlabel("False Alarm")
    plt.ylabel("Detection")

    return y_predict_prob, fpr, tpr, thresholds, auc

def get_file_dataframe(root, pathrows, bands):
    df = pd.DataFrame(columns=['p', 'pathrow', 'band', 'path_file'])
    for file in os.listdir(root):
        p = file.split("_")[4][1:]
        pathrow = file.split("_")[5]
        band = file.split("_")[6]

        if (pathrow in pathrows) and (band in bands):
            df = df.append(pd.Series({'p' : p,
                                      'pathrow' : pathrow,
                                      'band' : band,
                                      'path_file' : os.path.join(root, file)}),
                           ignore_index=True)
    return df

def merge_and_concat_file_dataframe(df_file):
    mergeon = ['pathrow', 'row', 'col', 'new_polygon_id', 'PLANT_PROVINCE_CODE',
               'PLANT_AMPHUR_CODE', 'PLANT_TAMBON_CODE', 'final_plant_date',
               'START_DATE', 'loss_ratio']
    list_df_p_pathrow = []
    for (p, pathrow), df_file_grp in df_file.groupby(["p", "pathrow"]):
        # Combine all bands in selected p and pathrow
        for index in df_file_grp.index:
            band = df_file_grp.loc[index, "band"]
            path_file = df_file_grp.loc[index, "path_file"]

            df = pd.read_parquet(path_file)
            df.columns = mergeon + [f"{band}_"+item for item in df.columns[-4:]]

            df = df[(df[df.columns[len(mergeon):]] != 0).all(axis=1)]
            if band == 'BQA':
                df = df[(df[df.columns[len(mergeon):]] != 1).all(axis=1)]

            if "df_p_pathrow" not in locals():
                df_p_pathrow = df

            else:
                df_p_pathrow = pd.merge(df_p_pathrow, df, how='inner', on=mergeon)

        # columns length must be equal to 10 + 20 (mergeon column + 4*num_band)
        if len(df_p_pathrow.columns) == (10+(4*len(pd.unique(df_file['band'])))):
            list_df_p_pathrow.append(df_p_pathrow)
        del df_p_pathrow

    df = pd.concat(list_df_p_pathrow, ignore_index=True)
    df.index = df['new_polygon_id']

    del list_df_p_pathrow
    df = df.drop_duplicates(subset=df.columns[len(mergeon):])
    return df
#%%
root_flood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_flood_value"
root_noflood = r"F:\CROP-PIER\CROP-WORK\landsat8_dataframe\ls8_noflood_value"
root_save = r"F:\CROP-PIER\CROP-WORK\Flood_evaluation\Landsat-8"
pathrows = ['130048', '130049', '129048', '129049']
#bands = ['B2', 'B3', 'B4', 'B5', 'BQA']
#pathrows = ['130048']
bands = ['B2', 'B3', 'B4', 'B5']
bands_combination = True
pathrows_combination = False
test_size = 0.20
random_state = 112
#%% Get all combination of bands ex. ['B2', 'B3'] >> [['B2'], ['B3'], ['B2', 'B3']]
if bands_combination:
    all_bands = []
    for r in range(len(bands) + 1):
        combinations_object = itertools.combinations(bands, r)
        combinations_list = list(combinations_object)
        all_bands += combinations_list
    all_bands = all_bands[1:]
#%% Get all combination of pathrows
if pathrows_combination:
    all_pathrows = []
    for r in range(len(pathrows) + 1):
        combinations_object = itertools.combinations(pathrows, r)
        combinations_list = list(combinations_object)
        all_pathrows += combinations_list
#%%
for bands in tqdm(all_bands):
#    for pathrows in tqdm(all_pathrows):
    if len(pathrows) == 0 or len(bands) == 0:
        continue
    tqdm.write(f"{list(bands)}, {list(pathrows)}")

    # Define save path
    path_save_folder = os.path.join(root_save, f"model_{'_'.join(pathrows)}_{'_'.join(bands)}")
    os.makedirs(path_save_folder, exist_ok=True)
    path_save_model = os.path.join(path_save_folder, 'model.joblib')
    path_save_plot = os.path.join(path_save_folder, 'ROC.png')
    path_save_min = os.path.join(path_save_folder, 'min_value.npy')
    path_save_max = os.path.join(path_save_folder, 'max_value.npy')
    path_save_df_stats = os.path.join(path_save_folder, 'df_stats.csv')
    path_save_roc_parameters = os.path.join(path_save_folder, 'roc_params.h5')
    
    if os.path.exists(path_save_model) and os.path.exists(path_save_plot) and os.path.exists(path_save_df_stats) and os.path.exists(path_save_roc_parameters):
        if 'BQA' in bands:
            if os.path.exists(path_save_min) and os.path.exists(path_save_max):
                tqdm.write(f"{list(bands)}, {list(pathrows)} already exists, skip")
                continue
        else:
            tqdm.write(f"{list(bands)}, {list(pathrows)} already exists, skip")
            continue

    # Get dataframe of file path of selecteh pathrow and band
    df_file_flood = get_file_dataframe(root_flood, pathrows, bands)
    df_file_noflood = get_file_dataframe(root_noflood, pathrows, bands)

    # Load data from dataframe of file path
    df_flood = merge_and_concat_file_dataframe(df_file_flood)
    df_noflood = merge_and_concat_file_dataframe(df_file_noflood)

    # For flood dataframe, keep only loss ratio in range [0.8, 1]
    df_flood = df_flood[(df_flood['loss_ratio'] >= 0.8) & (df_flood['loss_ratio'] <= 1)]
    columns_data = df_flood.columns[10:]

    # Normailize band QA (x-min)/(max-min) where max from max(flood, noflood) and min from min(flood, noflood)
    # Store only min and max first then normalize after sampling data
    if 'BQA' in bands:
        QA_max_value = max(df_noflood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']].max().max(),
                           df_flood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']].max().max())
        QA_min_value = min(df_noflood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']].min().min(),
                           df_flood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']].min().min())

    # Create train, test dataset, doesn't have dev set
    # Step1: Sampling noflood dataframe with the same length as flood dataframe (for balanced dataset no.'0' == no.'1')
    sample_size = len(df_flood)
    samples_noflood = df_noflood.sample(n=sample_size)
    samples_flood = df_flood.sample(n=sample_size)

    # Step1.1: Normalize band "QA" using min, max
    if 'BQA' in bands:
        samples_flood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']] = (samples_flood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']] - QA_min_value)/(QA_max_value-QA_min_value)
        samples_noflood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']] = (samples_noflood[['BQA_t-2', 'BQA_t-1', 'BQA_t+0', 'BQA_t+1']] - QA_min_value)/(QA_max_value-QA_min_value)

    # Step2: Concat flood and no flood into 'x' and label y then divide into train, test
    x = np.concatenate((samples_flood[columns_data].values, samples_noflood[columns_data].values), axis=0)
    y = np.zeros(len(x))
    y[:sample_size] = 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Define model and train
    model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, random_state=0, verbose=0, n_jobs=-1)
    model.fit(x_train, y_train)
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)

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

    # Create stats dataframe
    df_stats = pd.Series({"len(df_flood)":len(df_flood),
                          "len(df_noflood)":len(df_noflood),
                          "len(samples_flood)":len(samples_flood),
                          "len(samples_noflood)":len(samples_noflood),
                          "len(x_train)":len(x_train),
                          "len(x_test)":len(x_test),
                          "test_size":test_size,
                          "random_state":random_state,
                          "score_train":score_train,
                          "score_test":score_test,
                          "auc_train":auc_train,
                          "auc_test":auc_test,
                         })
    flood_counts = samples_flood['START_DATE'].dt.year.value_counts(sort=False)
    flood_counts.index = [f"{year}_flood" for year in flood_counts.index]
    noflood_counts = samples_noflood['START_DATE'].dt.year.value_counts(sort=False)
    noflood_counts.index = [f"{year}_noflood" for year in noflood_counts.index]
    df_stats = df_stats.append([flood_counts, noflood_counts])

    # Save figure, model, df_stats, roc params, max, min and free memory
    plt.savefig(path_save_plot, transparent=True, dpi=200)
    icedumpy.io_tools.save_model(model, path_save_model)
    df_stats.to_csv(path_save_df_stats, header=False)
    icedumpy.io_tools.save_h5(path=path_save_roc_parameters, dict_save={'fpr_train':fpr_train,'fpr_test':fpr_test,
                                                                        'tpr_train':tpr_train,'tpr_test':tpr_test,
                                                                        'thresholds_train':thresholds_train,'thresholds_test':thresholds_test,
                                                                       })
    
    if 'BQA' in bands: 
        np.save(path_save_max, QA_max_value)
        np.save(path_save_min, QA_min_value)
    
    del df_flood, df_noflood, samples_flood, samples_noflood, x_train, x_test, y_train, y_test, model
#%%
