import os
import matplotlib.pyplot as plt
import numpy as np
import flood_fn
#import icedumpy
#%%
index_names = [101, 102, 103, 104, 105, 106, 107, 108, 109, 
               201, 202, 203, 204, 205, 206, 207, 208,
               301, 302, 303, 304, 305, 306,
               401, 402, 403]
index_names = [f's{item}' for item in index_names]
root_df_dry = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_noflood_pixel_from_mapping_v3_nine_datapoint"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_flood_pixel_from_mapping_v3_nine_datapoint"
root_model = r"F:\CROP-PIER\CROP-WORK\Model_visualization\model_204_9"
root_save = rf'C:\Users\PongporC\Desktop\30-7-2020\sentinel-1\nine'
#%%
# Load data
for index_name in index_names:
    
    os.makedirs(root_save, exist_ok=True)
    df_dry, df_flood = flood_fn.load_dataset_df(root_df_dry, root_df_flood, index_name)
    columns = []
    for column in df_dry.columns:
        if 't(' in column:
            if int(column.split('_')[-1]) == 5:
                columns.append(column)
    print(df_dry[columns].mean().values)
    print(df_flood[columns].mean().values)
    #%%
    # Train model
    model, [x_train, x_test, y_train, y_test] = flood_fn.train_model(df_dry, df_flood, balanced=True, verbose=0)
    #%%
    # Show ROC curve of train and test
    plt.figure(figsize=(16.0, 9.0))
    y_predict_prob_train, fpr_train, tpr_train, thresholds_train = flood_fn.plot_roc_curve(model, x_train, y_train, label = 'train', color='r-')
    y_predict_prob_test, fpr_test, tpr_test, thresholds_test = flood_fn.plot_roc_curve(model, x_test, y_test, label = 'test ', color='b-')
    plt.xticks(np.arange(-0.05, 1.05, 0.05))
    plt.yticks(np.arange(0, 1., 0.1))
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.grid()
    plt.legend(loc = 4)
    plt.title(f"Scene: {index_name[1:]}, Train samples: {len(x_train)}, Test samples: {len(x_test)}")

#%%
#flood_fn.save_model(model, os.path.join(root_model, "s204.joblib"))
    plt.savefig(os.path.join(root_save, f"ROC_{index_name}.png"), transparent=True, dpi=200)
    plt.close('all')
    del model, df_dry, df_flood, x_train, x_test, y_train, y_test
#icedumpy.io_tools.save_h5(path=os.path.join(root_model, "ROC_params.h5"), dict_save={'y_predict_prob_train':y_predict_prob_train, 'y_predict_prob_test':y_predict_prob_test,
#                                                                                    'fpr_train':fpr_train,'fpr_test':fpr_test,
#                                                                                    'tpr_train':tpr_train,'tpr_test':tpr_test,
#                                                                                    'thresholds_train':thresholds_train,'thresholds_test':thresholds_test,
#                                                                                    })