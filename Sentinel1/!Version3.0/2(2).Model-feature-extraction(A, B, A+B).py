import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from icedumpy.plot_tools import plot_roc_curve, set_roc_plot_template
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe, load_mapping
#%% Define functions
def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]

    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df

def extract_df_flood_nonflood_features(root_df_flood, root_df_nonflood, p, strip_id):
    df_flood = load_s1_flood_nonflood_dataframe(root_df_flood, p, strip_id=strip_id)
    df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p, strip_id=strip_id)

    # Replace negative data with nan
    df_flood = replace_zero_with_nan(df_flood)
    df_nonflood = replace_zero_with_nan(df_nonflood)

    # Drop row with nan (any)
    columns_pixel_values = df_flood.columns[df_flood.columns.str.contains(r'^t\d+$')].tolist()
    df_flood = df_flood.dropna(subset=columns_pixel_values)
    df_nonflood = df_nonflood.dropna(subset=columns_pixel_values)

    # df_flood = df_flood.loc[~pd.isna(df_flood.loc[:, columns_pixel_values]).all(axis=1)]
    # df_nonflood = df_nonflood.loc[~pd.isna(df_nonflood.loc[:, columns_pixel_values]).all(axis=1)]

    # # Sampling df_nonflood (equal ext_act_id as df_flood)
    # df_nonflood_unique = df_nonflood["ext_act_id"].unique()
    # df_nonflood = df_nonflood[df_nonflood["ext_act_id"].isin(df_nonflood_unique[np.random.permutation(len(df_nonflood_unique))[:len(df_flood["ext_act_id"].unique())]])]

    # Create diff columns
    df_flood = pd.concat([df_flood, df_flood[columns_pixel_values].diff(axis=1).iloc[:, 1:].rename(columns={f"{columns_pixel_values[i+1]}":f"diff_{columns_pixel_values[i+1]}-{columns_pixel_values[i]}" for i in range(0, len(columns_pixel_values)-1)})], axis=1)
    df_nonflood = pd.concat([df_nonflood, df_nonflood[columns_pixel_values].diff(axis=1).iloc[:, 1:].rename(columns={f"{columns_pixel_values[i+1]}":f"diff_{columns_pixel_values[i+1]}-{columns_pixel_values[i]}" for i in range(0, len(columns_pixel_values)-1)})], axis=1)
    columns_pixel_diff_values = df_flood.columns[df_flood.columns.str.match(r"^diff_")]
    
    # Min, max, averagePixel values (First half and Second half)
    df_flood = df_flood.assign(min_pixel_value_first_half=df_flood[columns_pixel_values[:len(columns_pixel_values)//2]].min(axis=1))
    df_flood = df_flood.assign(max_pixel_value_first_half=df_flood[columns_pixel_values[:len(columns_pixel_values)//2]].max(axis=1))
    df_flood = df_flood.assign(avg_pixel_value_first_half=df_flood[columns_pixel_values[:len(columns_pixel_values)//2]].mean(axis=1))
    df_flood = df_flood.assign(min_pixel_value_second_half=df_flood[columns_pixel_values[len(columns_pixel_values)//2:]].min(axis=1))
    df_flood = df_flood.assign(max_pixel_value_second_half=df_flood[columns_pixel_values[len(columns_pixel_values)//2:]].max(axis=1))
    df_flood = df_flood.assign(avg_pixel_value_second_half=df_flood[columns_pixel_values[len(columns_pixel_values)//2:]].mean(axis=1))
    
    df_nonflood = df_nonflood.assign(min_pixel_value_first_half=df_nonflood[columns_pixel_values[:len(columns_pixel_values)//2]].min(axis=1))
    df_nonflood = df_nonflood.assign(max_pixel_value_first_half=df_nonflood[columns_pixel_values[:len(columns_pixel_values)//2]].max(axis=1))
    df_nonflood = df_nonflood.assign(avg_pixel_value_first_half=df_nonflood[columns_pixel_values[:len(columns_pixel_values)//2]].mean(axis=1))
    df_nonflood = df_nonflood.assign(min_pixel_value_second_half=df_nonflood[columns_pixel_values[len(columns_pixel_values)//2:]].min(axis=1))
    df_nonflood = df_nonflood.assign(max_pixel_value_second_half=df_nonflood[columns_pixel_values[len(columns_pixel_values)//2:]].max(axis=1))
    df_nonflood = df_nonflood.assign(avg_pixel_value_second_half=df_nonflood[columns_pixel_values[len(columns_pixel_values)//2:]].mean(axis=1))
    
    # Min, max, average Diff values (First half and Second half)
    df_flood = df_flood.assign(min_pixel_diff_value_first_half=df_flood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].min(axis=1))
    df_flood = df_flood.assign(max_pixel_diff_value_first_half=df_flood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].max(axis=1))
    df_flood = df_flood.assign(avg_pixel_diff_value_first_half=df_flood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].mean(axis=1))
    df_flood = df_flood.assign(min_pixel_diff_value_second_half=df_flood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].min(axis=1))
    df_flood = df_flood.assign(max_pixel_diff_value_second_half=df_flood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].max(axis=1))
    df_flood = df_flood.assign(avg_pixel_diff_value_second_half=df_flood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].mean(axis=1))
    
    df_nonflood = df_nonflood.assign(min_pixel_diff_value_first_half=df_nonflood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].min(axis=1))
    df_nonflood = df_nonflood.assign(max_pixel_diff_value_first_half=df_nonflood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].max(axis=1))
    df_nonflood = df_nonflood.assign(avg_pixel_diff_value_first_half=df_nonflood[columns_pixel_diff_values[:len(columns_pixel_diff_values)//2]].mean(axis=1))
    df_nonflood = df_nonflood.assign(min_pixel_diff_value_second_half=df_nonflood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].min(axis=1))
    df_nonflood = df_nonflood.assign(max_pixel_diff_value_second_half=df_nonflood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].max(axis=1))
    df_nonflood = df_nonflood.assign(avg_pixel_diff_value_second_half=df_nonflood[columns_pixel_diff_values[len(columns_pixel_diff_values)//2:]].mean(axis=1))
    
    df_flood.drop(columns=columns_pixel_values, inplace=True)
    df_nonflood.drop(columns=columns_pixel_values, inplace=True)
    df_flood.drop(columns=columns_pixel_diff_values, inplace=True)
    df_nonflood.drop(columns=columns_pixel_diff_values, inplace=True)
    return df_flood, df_nonflood

def get_threshold_of_selected_fpr(fpr, thresholds, selected_fpr):
    index = np.argmin(np.abs(fpr - selected_fpr))
    return thresholds[index]

def histsubplot(df, group_name, column, ax, color="salmon", bins=20):
    sns.histplot(df.loc[df["Group"] == group_name, column],
                 kde=True, color=color, bins=bins, ax=ax)
    ax.set_title(group_name)
#%% Define directories
root_df_mapping = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1_polygon_id_rowcol_map_prov_scene_v4(at-False)"
root_save_plot = r"F:\CROP-PIER\CROP-WORK\Presentation\20200908\Plot"
#%%  Define parameters
list_strip_id = ["302", "303", "304", "305", "306", "401", "402", "403"]
sat_type = "S1AB"
p = None
#%% Define parameters
# strip_id = "401"
for strip_id in list_strip_id:
    root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
    root_df_nonflood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"

    # Add strip_id to path
    root_df_flood = os.path.join(root_df_flood, f"{sat_type.lower()}_flood_pixel")
    root_df_nonflood = os.path.join(root_df_nonflood, f"{sat_type.lower()}_nonflood_pixel")
    os.makedirs(os.path.join(root_save_plot, sat_type, strip_id, "new_features"), exist_ok=True)
    #%% Load dfflood, df_nonflood and drop nan
    df_flood, df_nonflood = extract_df_flood_nonflood_features(root_df_flood, root_df_nonflood, p, strip_id)
    columns_feature_values = df_flood.columns[df_flood.columns.str.contains("min") | df_flood.columns.str.contains("max") |df_flood.columns.str.contains("avg")]
    #%%
    # all_touched = False 
    # tier = [1]
    for tier, all_touched in zip([[1, 2], [1, 2], [1]], [True, False, False]):
        print(strip_id, tier, all_touched)
        # =============================================================================
        # Create train, test sample data
        # =============================================================================
        df_sample = pd.concat([df_flood, df_nonflood])
        if sat_type == "S1A":
            df_sample = df_sample.loc[df_sample["ANCHOR_DATE"] >= datetime.datetime(2018, 6, 1)]
        
        if (all_touched == False):
            df_mapping, _ = load_mapping(root_df_mapping, strip_id=strip_id, list_p=list(map(str, df_sample["PLANT_PROVINCE_CODE"].unique())))
            df_sample = pd.merge(df_sample, df_mapping, how='inner', on=["new_polygon_id", "row", "col"], left_index=True)

        df_sample = df_sample.assign(label=(df_sample["loss_ratio"] != 0).astype("uint8"))
        # df_sample = df_sample.sample(frac=1.0)
        df_sample_drop_dup = df_sample.drop_duplicates(subset=columns_feature_values)

        if (all_touched == False) and (len(tier) != 2):
            df_sample_drop_dup = df_sample_drop_dup[df_sample_drop_dup["tier"].isin(tier)]
            
        df_sample_drop_dup = df_sample_drop_dup.loc[(df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"].between(0.8, 1.0))]
        df_sample_drop_dup = df_sample_drop_dup[(df_sample_drop_dup["ext_act_id"].isin(np.random.choice(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 0, "ext_act_id"].unique(), len(df_sample_drop_dup.loc[df_sample_drop_dup["label"] == 1, "ext_act_id"].unique()), replace=False))) | (df_sample_drop_dup["label"] == 1)]
        df_sample_drop_dup = df_sample_drop_dup.sample(frac=1.0)
        
        # =============================================================================
        # Create train-test samples
        # =============================================================================
        df_train = df_sample_drop_dup.loc[(~(df_sample_drop_dup["ext_act_id"]%10).isin([8, 9])) & ((df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"] >= 0.8))]
        df_test = df_sample_drop_dup.loc[((df_sample_drop_dup["ext_act_id"]%10).isin([8, 9])) & ((df_sample_drop_dup["loss_ratio"] == 0) | (df_sample_drop_dup["loss_ratio"] >= 0.8))]
        
        if (len(df_train) == 0) or (len(df_test) == 0):
            continue
        # =============================================================================
        # Fit model
        # =============================================================================
        model = RandomForestClassifier(min_samples_leaf=5, max_depth=10, min_samples_split=10,
                                       verbose=0, class_weight="balanced",
                                       n_jobs=-1, random_state=42)
        model.fit(df_train[columns_feature_values].values, df_train["label"].values)

        # =============================================================================
        # Plot ROC
        # =============================================================================
        plt.close('all')
        fig, ax = plt.subplots(figsize=(16, 9))
        ax, y_predict_prob, fpr, tpr, thresholds, auc = plot_roc_curve(model, df_train[columns_feature_values].values, df_train["label"].values, "train", color="g", ax=ax)
        ax, _, _, _, _, _ = plot_roc_curve(model, df_test[columns_feature_values].values, df_test["label"].values, "test", color='b', ax=ax)
        ax = set_roc_plot_template(ax)
        ax.set_title(f'ROC Curve: {strip_id}\nAll_touched({all_touched}), Tier{tuple(tier)}\nTrain samples: Flood:{(df_train["label"] == 1).sum():,}, Non-Flood:{(df_train["label"] == 0).sum():,}\nTest samples: Flood:{(df_test["label"] == 1).sum():,}, Non-Flood:{(df_test["label"] == 0).sum():,}')
        fig.savefig(os.path.join(root_save_plot, sat_type, strip_id, "new_features", f"{strip_id}_ROC_at({all_touched})_tier{tuple(tier)}.png"))
        plt.close('all')

        # =============================================================================
        # Prepare data for visualization
        # =============================================================================
        threshold = get_threshold_of_selected_fpr(fpr, thresholds, 0.1)
        df_test = df_sample.loc[(df_sample["ext_act_id"]%10).isin([8, 9])]
        if (all_touched == False) and (len(tier) != 2):
            df_test = df_test[df_test["tier"].isin(tier)]

        # Mean of loss probabilities
        df_test = df_test.assign(pred_prob = model.predict_proba(df_test[columns_feature_values])[:, 1])

        # Mean of pixel-wise classification
        df_test = df_test.assign(pred = (model.predict_proba(df_test[columns_feature_values])[:, 1] >= threshold).astype("uint8"))

        df_test = df_test.groupby(["ext_act_id", "loss_ratio"]).agg("mean").rename(columns={"pred_prob":"pred_prob_mean", "pred":"pred_mean"}).reset_index()
        df_test.loc[df_test["loss_ratio"] == 0, "Group"] = "Actual loss ratio = 0"
        df_test.loc[(df_test["loss_ratio"] > 0) & (df_test["loss_ratio"] < 0.8), "Group"] = "0 < Actual loss ratio < 0.8"
        df_test.loc[(df_test["loss_ratio"] >= 0.8) & (df_test["loss_ratio"] <= 1.0), "Group"] = "0.8 <= Actual loss ratio <= 1.0"
        df_test["Mean of loss probabilities - Actual loss ratio"] = df_test["pred_prob_mean"] - df_test["loss_ratio"]
        df_test["Mean of pixel-wise classification - Actual loss ratio"] = df_test["pred_mean"] - df_test["loss_ratio"]

        # =============================================================================
        # Data visualization#1: Plot Hist2D (Mean of loss proabilities)
        # =============================================================================
        g = sns.jointplot(data=df_test, x="pred_prob_mean", y="loss_ratio",
                          height=9, marginal_kws={"bins":50},
                          marginal_ticks=True, kind='hist', bins=50,
                          cmap="light:b",
                          xlim=(0, 1),
                          ylim=(0, 1))
        g.ax_joint.set_xticks(np.linspace(0, 1.00, 11))
        g.ax_joint.set_yticks(np.linspace(0, 1.00, 11))
        g.ax_joint.set_xlabel("Mean of loss probabilities")
        g.ax_joint.set_ylabel("Actual loss ratio")
        plt.suptitle(f"Heatmap: {strip_id}\nAll_touched({all_touched}), Tier{tuple(tier)}")
        g.fig.savefig(os.path.join(root_save_plot, sat_type, strip_id, "new_features", f"{strip_id}_heatmap_mean_loss_prob_at({all_touched})_tier{tuple(tier)}.png"))
        plt.close('all')

        # =============================================================================
        # Data visualization#2: Plot Hist2D (Mean of pixel-wise classification)
        # =============================================================================
        g = sns.jointplot(data=df_test, x="pred_mean", y="loss_ratio",
                          height=9, marginal_kws={"bins":50},
                          marginal_ticks=True, kind='hist', bins=50,
                          cmap="light:b",
                          xlim=(0, 1),
                          ylim=(0, 1))
        g.ax_joint.set_xticks(np.linspace(0, 1.00, 11))
        g.ax_joint.set_yticks(np.linspace(0, 1.00, 11))
        g.ax_joint.set_xlabel("Mean of pixel-wise classification")
        g.ax_joint.set_ylabel("Actual loss ratio")
        plt.suptitle(f"Heatmap: {strip_id}\nAll_touched({all_touched}), Tier{tuple(tier)}")
        g.fig.savefig(os.path.join(root_save_plot, sat_type, strip_id, "new_features", f"{strip_id}_heatmap_mean_pixel_at({all_touched})_tier{tuple(tier)}.png"))
        plt.close('all')
        # =============================================================================
        # Data visualization#3: Plot diff histograms
        # =============================================================================

        fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=False)
        ax[0, 0].set_xlim(-1, 1)
        
        histsubplot(df_test, "Actual loss ratio = 0", "Mean of loss probabilities - Actual loss ratio", ax[0, 0], color='salmon')
        ax[0, 0].set_title(f"Actual loss ratio = 0 (Total samples: {len(df_test.loc[df_test['Group'] == 'Actual loss ratio = 0']):,})")
        
        histsubplot(df_test, "0 < Actual loss ratio < 0.8", "Mean of loss probabilities - Actual loss ratio", ax[1, 0], color='salmon')
        ax[1, 0].set_title(f"0 < Actual loss ratio < 0.8 (Total samples: {len(df_test.loc[df_test['Group'] == '0 < Actual loss ratio < 0.8']):,})")

        histsubplot(df_test, "0.8 <= Actual loss ratio <= 1.0", "Mean of loss probabilities - Actual loss ratio", ax[2, 0], color='salmon')
        ax[2, 0].set_title(f"0.8 <= Actual loss ratio <= 1.0 (Total samples: {len(df_test.loc[df_test['Group'] == '0.8 <= Actual loss ratio <= 1.0']):,})")
        
        histsubplot(df_test, "Actual loss ratio = 0", "Mean of pixel-wise classification - Actual loss ratio", ax[0, 1], color='lime')
        ax[0, 1].set_title(f"Actual loss ratio = 0 (Total samples: {len(df_test.loc[df_test['Group'] == 'Actual loss ratio = 0']):,})")
        
        histsubplot(df_test, "0 < Actual loss ratio < 0.8" , "Mean of pixel-wise classification - Actual loss ratio", ax[1, 1], color='lime')
        ax[1, 1].set_title(f"0 < Actual loss ratio < 0.8 (Total samples: {len(df_test.loc[df_test['Group'] == '0 < Actual loss ratio < 0.8']):,})")
        
        histsubplot(df_test, "0.8 <= Actual loss ratio <= 1.0", "Mean of pixel-wise classification - Actual loss ratio", ax[2, 1], color='lime')
        ax[2, 1].set_title(f"0.8 <= Actual loss ratio <= 1.0 (Total samples: {len(df_test.loc[df_test['Group'] == '0.8 <= Actual loss ratio <= 1.0']):,})")
        
        fig.suptitle(f"Histogram: {strip_id}\nAll_touched({all_touched}), Tier{tuple(tier)}")
        fig.savefig(os.path.join(root_save_plot, sat_type, strip_id, "new_features", f"{strip_id}_diff_hist_at({all_touched})_tier{tuple(tier)}.png"))
        plt.close('all')
        
        del df_sample, df_sample_drop_dup, df_test, df_train
#%%