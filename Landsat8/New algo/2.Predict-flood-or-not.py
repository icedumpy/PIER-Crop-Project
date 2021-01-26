import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from icedumpy.df_tools import load_mapping, load_vew, clean_and_process_vew, set_index_for_loc
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.io_tools import save_model, save_h5
#%% Global variables
ROOT_RASTER = r"G:\!PIER\!FROM_2TB\LS8-MIN-FROM-VRT"
ROOT_DF_MAPPING = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_polygon_id_rowcol_map_prov_scene_merged_v2"
ROOT_DF_VEW = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
ROOT_SAVE = r"F:\CROP-PIER\CROP-WORK\Model_visualization\model_min"
YEARS = [2017, 2018, 2019]
#%%
def load_mapping_vew(ROOT_DF_MAPPING, ROOT_DF_VEW, pathrow):
    # Load mapping of selected pathrow
    # Load vew following mapping (p(s) in selected pathrow)
    df_mapping, list_p = load_mapping(ROOT_DF_MAPPING, list_pathrow=[pathrow])
    df_mapping = set_index_for_loc(df_mapping, column='new_polygon_id')

    df_vew = load_vew(ROOT_DF_VEW, list_p)
    df_vew = clean_and_process_vew(df_vew, list_new_polygon_id=df_mapping['new_polygon_id'].unique())

    return df_mapping, df_vew

def separate_vew_flood_nonflood_by_year(df_vew, year):
    # Separate flood and nonflood from vew
    # and filter by year
    df_vew = df_vew.loc[df_vew['final_plant_date'].dt.year == year]

    df_vew_flood = df_vew.loc[(df_vew["START_DATE"].dt.year == year) & (df_vew['loss_ratio'] >= 0.8)]
    df_vew_nonflood = df_vew.loc[pd.isna(df_vew["DANGER_TYPE_NAME"]) & (df_vew['final_plant_date'].dt.year == year)]

    assert len(df_vew_flood["DANGER_TYPE_NAME"].unique()) == 1 # Make sure only flood
    assert len(df_vew_flood["START_DATE"].dt.year.unique()) == 1 # Make sure flood has only one year
    assert df_vew_flood["START_DATE"].dt.year.unique() == year # Make sure flood year is correct

    assert df_vew_nonflood["DANGER_TYPE_NAME"].unique() == None # Makse sure only normal in non-flood
    assert len(df_vew_nonflood["final_plant_date"].dt.year.unique()) == 1 # Make sure non-flood has only one year
    assert df_vew_nonflood["final_plant_date"].dt.year.unique() == year # Make sure non-flood year is correct

    return df_vew_flood, df_vew_nonflood

def get_mapping_rows_cols_from_new_polygon_id(df_mapping, list_new_polygon_id):
    # Get rows and columns of new_polygon_id(s)
    df_mapping_selected = df_mapping.loc[df_mapping['new_polygon_id'].isin(list_new_polygon_id), ['row', 'col']]

    return df_mapping_selected

def drop_duplicate_row_col_from_flood_nonflood(rows_cols_flood, rows_cols_nonflood):
    # Drop duplicate
    rows_cols_flood["from"] = "flood"
    rows_cols_nonflood["from"] = "nonflood"

    # If duplicate, keep only row, col of flood
    rows_cols_drop_duplicate = pd.concat([rows_cols_flood, rows_cols_nonflood]).drop_duplicates(subset=['row', 'col'], keep='first')

    return (rows_cols_drop_duplicate.loc[rows_cols_drop_duplicate["from"] == 'flood', ['row', 'col']],
            rows_cols_drop_duplicate.loc[rows_cols_drop_duplicate["from"] == 'nonflood', ['row', 'col']])

def get_flood_nonflood_pixel_values(ROOT_RASTER, pathrow, year, rows_cols_flood, rows_cols_nonflood):
    raster = rasterio.open(os.path.join(ROOT_RASTER, pathrow, f"{year}_min"))
    raster_img = raster.read()
    raster_img = np.moveaxis(raster_img, 0, -1)

    pixel_value_flood = raster_img[rows_cols_flood['row'].values, rows_cols_flood['col'].values]
    pixel_value_nonflood = raster_img[rows_cols_nonflood['row'].values, rows_cols_nonflood['col'].values]

    pixel_value_flood = pixel_value_flood[(pixel_value_flood != 0).all(axis=1)]
    pixel_value_nonflood = pixel_value_nonflood[(pixel_value_nonflood != 0).all(axis=1)]

    return pixel_value_flood, pixel_value_nonflood

def concat_split_train(pixel_value_flood, pixel_value_nonflood):
    X = np.vstack((pixel_value_flood,
                   pixel_value_nonflood))
    Y = np.concatenate((np.ones((len(pixel_value_flood)), dtype=np.uint8),
                        np.zeros((len(pixel_value_nonflood)), dtype=np.uint8)))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, random_state=0, verbose=0, n_jobs=-1)
    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test

def main(pathrow):
    folder_save = os.path.join(ROOT_SAVE, pathrow)
    os.makedirs(folder_save, exist_ok=True)

    print("LOAD DATA: df_mapping, df_vew")
    # Load mappings and activities of the selected pathrow
    df_mapping, df_vew = load_mapping_vew(ROOT_DF_MAPPING, ROOT_DF_VEW, pathrow)
    
    list_pixel_value_flood = []
    list_pixel_value_nonflood = []
    
    # Loop for eacg year [2017, 2018, 2019]
    for year in YEARS:
        print("COLLECTING DATA FROM YEAR:", year)
        df_vew_flood, df_vew_nonflood = separate_vew_flood_nonflood_by_year(df_vew, year)
        
        # Get rows and columns of flood
        rows_cols_flood = get_mapping_rows_cols_from_new_polygon_id(df_mapping, df_vew_flood['new_polygon_id'].unique())
        # Get rows and columns of non flood
        rows_cols_nonflood = get_mapping_rows_cols_from_new_polygon_id(df_mapping, df_vew_nonflood['new_polygon_id'].unique())
        
        # Drop duplicate rows and columns
        rows_cols_flood, rows_cols_nonflood = drop_duplicate_row_col_from_flood_nonflood(rows_cols_flood, rows_cols_nonflood)

        pixel_value_flood, pixel_value_nonflood = get_flood_nonflood_pixel_values(ROOT_RASTER, pathrow, year, rows_cols_flood, rows_cols_nonflood)
        pixel_value_nonflood = pixel_value_nonflood[np.random.permutation(len(pixel_value_flood))]

        list_pixel_value_flood.append(pixel_value_flood)
        list_pixel_value_nonflood.append(pixel_value_nonflood)
        
    pixel_value_flood = np.concatenate(list_pixel_value_flood)
    pixel_value_nonflood = np.concatenate(list_pixel_value_nonflood)

    print("START TRAINING")
    model, x_train, x_test, y_train, y_test = concat_split_train(pixel_value_flood, pixel_value_nonflood)

    print("PLOT ROC CURVE")
    # Plot ROC Curve of train and test
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(16.0, 9.0))
    ax, y_predict_prob_train, fpr_train, tpr_train, thresholds_train, auc_train = plot_roc_curve(ax, model, x_train, y_train, label='train', color='r-')
    ax, y_predict_prob_test, fpr_test, tpr_test, thresholds_test, auc_test = plot_roc_curve(ax, model, x_test, y_test, label='test', color='b-')
    ax = set_roc_plot_template(ax)

    print("SAVE RESULTS")
    fig.savefig(os.path.join(folder_save, "ROC.png"), transparent=True, dpi=200)
    save_model(os.path.join(folder_save, "model.joblib"), model)
    save_h5(os.path.join(folder_save, "model_params.h5"), dict_save={'y_predict_prob_train':y_predict_prob_train,
                                                                     'y_predict_prob_test':y_predict_prob_test,
                                                                     'fpr_train':fpr_train, 'fpr_test':fpr_test,
                                                                     'tpr_train':tpr_train, 'tpr_test':tpr_test,
                                                                     'thresholds_train':thresholds_train,
                                                                     'thresholds_test':thresholds_test,
                                                                     'auc_train':auc_train, 'auc_test':auc_test,
                                                                     'x_train':x_train, 'x_test':x_test,
                                                                     'y_train':y_train, 'y_test':y_test
                                                                    })
    print()
#%%
if __name__=='__main__':
    for pathrow in os.listdir(ROOT_RASTER):
        print("START PATHROW:", pathrow)
        main(pathrow)
