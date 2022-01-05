import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from imblearn.over_sampling import SMOTE 
from icedumpy.plot_tools import plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score, plot_confusion_matrix
#%%
def get_combinations(features):
    list_feature_combinations = []
    for i in range(1, len(features)+1):
        list_feature_combinations+=list(map(list, list(combinations(features, i))))
    return list_feature_combinations

def run_model(x_train, y_train, x_test, y_test, features, n_trials=10):
    list_f1 = []
    # Run model for n trails then find average "f1"
    for i in range(n_trials):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, criterion="gini", n_jobs=-1))
        ])
        pipeline.fit(x_train, y_train)
        
        # Get predicted
        y_test_pred  = pipeline.predict(x_test)
        
        # Calculate score
        f1 = f1_score(y_test, y_test_pred)
        
        # Append F1
        list_f1.append(f1)
    
    dict_report = {
        "features" : features,
        "f1" : sum(list_f1)/len(list_f1),
    }
    return dict_report

def compare_model(df_tambon_train, df_tambon_test, list_feature_combinations, n_trials=10):
    list_dict_report = list()
    for features in list_feature_combinations:
        # Train and evaluate model
        x_train, y_train = df_tambon_train[features].values, df_tambon_train["y"].values
        x_test,  y_test  = df_tambon_test[features].values, df_tambon_test["y"].values
        model_report = run_model(x_train, y_train, x_test, y_test, features, n_trials)
        # Append results to list
        list_dict_report.append(model_report)
    return list_dict_report

def make_report(list_dict_report):
    list_df = []
    for dict_report in list_dict_report:
        list_df.append(
            pd.DataFrame([
                dict_report["f1"],
            ], columns = ["&".join(dict_report["features"])], index=["f1"])
        )
    df_report = pd.concat(list_df, axis=1)
    return df_report

def plot_report(df_report, criteria, title=None, xlabels=None):
    # Make df_report easier to display
    df_report = df_report.loc[[criteria]].T
    if not xlabels is None:
        df_report = df_report.assign(xlabels=xlabels)
        df_report = df_report.set_index("xlabels")
    df_report = df_report.sort_values(by=criteria, ascending=False)
    df_report = df_report.iloc[:15]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 9))
    df_report.plot(kind="bar", rot=30, ax=ax)
    if not title is None:
        ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(color="black", linestyle="--", alpha=0.3)
    ax.bar_label(ax.containers[0], fmt="%.4f")
    ax.set_xlabel("")
    return fig, ax

def main_features_comparison(df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
                             folder_name, figure_name, report_name,
                             figure_xlabels=None, figure_title=None, n_trials=10):
    # Make reports
    list_dict_report = compare_model(df_tambon_train, df_tambon_test, list_feature_combinations, n_trials)
    df_report = make_report(list_dict_report)
    
    # Plot results
    plt.close("all")
    fig, ax = plot_report(
        df_report, 
        criteria=criteria,
        title=figure_title,
        xlabels=figure_xlabels
    )
    
    # Save both figure and report
    folder_save = os.path.join(root_save, folder_name)
    os.makedirs(folder_save, exist_ok=True)
    
    # Save figure
    fig.savefig(os.path.join(folder_save, figure_name), bbox_inches="tight")
    
    # Save Report
    df_report.columns = pd.MultiIndex.from_tuples([(short, real) for short, real in zip(figure_xlabels, df_report.columns)])
    df_report.to_csv(os.path.join(folder_save, report_name))    
    return df_report


def get_combinations(features):
    list_feature_combinations = []
    for i in range(1, len(features)+1):
        list_feature_combinations+=list(map(list, list(combinations(features, i))))
    return list_feature_combinations
#%%
# Save folder
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211220"
path_df_tambon = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_tambon.parquet"

# Load train test dataframe
df_list_train = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_train.csv").iloc[:, 1]
df_list_test  = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_test.csv").iloc[:, 1]

# Load tambon level features (if available)
df_tambon = pd.read_parquet(path_df_tambon)
#%% Separate train test
df_tambon_train = df_tambon[df_tambon["tambon_pcode"].isin(df_list_train)]
df_tambon_test  = df_tambon[df_tambon["tambon_pcode"].isin(df_list_test)]

# Upsampling minority class (Only training data)
columns_training_feature = [column for column in df_tambon.columns if column.startswith("x_")]
df_tambon_train = pd.concat(SMOTE(sampling_strategy="minority", random_state=42).fit_resample(df_tambon_train[columns_training_feature], df_tambon_train["y"]), axis=1)
#%%
n_trials = 10
criteria = "f1"
list_report_main = []
#%%
features_main = [
    'x_s1_bc_drop+bc_min_p5',
    'x_s1_bc_drop+bc_min_p10',
    'x_s1_bc_drop+bc_min_min',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_strict_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_strict_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_strict_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_strict_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_relax_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_relax_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_relax_max',
    'x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_relax_max',
    'x_gistda_flood_ratio_1-5_max',
    'x_gistda_flood_ratio_1-5_p75',
    'x_gistda_flood_ratio_1-5_p95',
    'x_gistda_flood_ratio_6-10_max',
    'x_gistda_flood_ratio_6-10_p75',
    'x_gistda_flood_ratio_6-10_p95',
    'x_gistda_flood_ratio_11-15_max',
    'x_gistda_flood_ratio_11-15_p75',
    'x_gistda_flood_ratio_11-15_p95',
    'x_gistda_flood_ratio_15+_max',
    'x_gistda_flood_ratio_15+_p75',
    'x_gistda_flood_ratio_15+_p95',
    'x_gsmap_rain_ph1_CR_max',
    'x_gsmap_rain_ph1_CR_p75',
    'x_gsmap_rain_ph1_CR_p95',
    'x_gsmap_rain_ph2_CR_max',
    'x_gsmap_rain_ph2_CR_p75',
    'x_gsmap_rain_ph2_CR_p95',
    'x_gsmap_rain_ph3_CR_max',
    'x_gsmap_rain_ph3_CR_p75',
    'x_gsmap_rain_ph3_CR_p95',
    'x_gsmap_rain_ph4_CR_max',
    'x_gsmap_rain_ph4_CR_p75',
    'x_gsmap_rain_ph4_CR_p95',
    'x_smap_soil_moist_pctl_max_sm_max',
    'x_smap_soil_moist_v2_cnsct_period_above_95_relax_max'
]
#%%
columns_hls = [column for column in df_tambon.columns if "hls_ndvi" in column]
columns_modis = [column for column in df_tambon.columns if "modis_ndvi" in column]
#%%
# =============================================================================
# HLS
# =============================================================================
list_feature_combinations = [
    # Whole season
    ['x_hls_ndvi_v2_max_whssn_max', 'x_hls_ndvi_v2_max_whssn_p75', 'x_hls_ndvi_v2_max_whssn_p90', 'x_hls_ndvi_v2_max_whssn_p95'],
    ['x_hls_ndvi_v2_min_whssn_min', 'x_hls_ndvi_v2_min_whssn_p5', 'x_hls_ndvi_v2_min_whssn_p10', 'x_hls_ndvi_v2_min_whssn_p25'],
    ['x_hls_ndvi_v2_pctl_max_whssn_max', 'x_hls_ndvi_v2_pctl_max_whssn_p75', 'x_hls_ndvi_v2_pctl_max_whssn_p90', 'x_hls_ndvi_v2_pctl_max_whssn_p95'],
    ['x_hls_ndvi_v2_pctl_min_whssn_min', 'x_hls_ndvi_v2_pctl_min_whssn_p5', 'x_hls_ndvi_v2_pctl_min_whssn_p10', 'x_hls_ndvi_v2_pctl_min_whssn_p25'],
    # Stage 2to3
    ['x_hls_ndvi_v2_max_stg2to3_max', 'x_hls_ndvi_v2_max_stg2to3_p75', 'x_hls_ndvi_v2_max_stg2to3_p90', 'x_hls_ndvi_v2_max_stg2to3_p95'],
    ['x_hls_ndvi_v2_min_stg2to3_min', 'x_hls_ndvi_v2_min_stg2to3_p5', 'x_hls_ndvi_v2_min_stg2to3_p10', 'x_hls_ndvi_v2_min_stg2to3_p25'],
    ['x_hls_ndvi_v2_pctl_max_stg2to3_max', 'x_hls_ndvi_v2_pctl_max_stg2to3_p75', 'x_hls_ndvi_v2_pctl_max_stg2to3_p90', 'x_hls_ndvi_v2_pctl_max_stg2to3_p95'],
    ['x_hls_ndvi_v2_pctl_min_stg2to3_min', 'x_hls_ndvi_v2_pctl_min_stg2to3_p5', 'x_hls_ndvi_v2_pctl_min_stg2to3_p10', 'x_hls_ndvi_v2_pctl_min_stg2to3_p25'],
    # Stage 1, 2, 3, 4
    # Max
    [
        'x_hls_ndvi_v2_max_stg1_max', 'x_hls_ndvi_v2_max_stg1_p75', 'x_hls_ndvi_v2_max_stg1_p90', 'x_hls_ndvi_v2_max_stg1_p95',
        'x_hls_ndvi_v2_max_stg2_max', 'x_hls_ndvi_v2_max_stg2_p75', 'x_hls_ndvi_v2_max_stg2_p90', 'x_hls_ndvi_v2_max_stg2_p95',
        'x_hls_ndvi_v2_max_stg3_max', 'x_hls_ndvi_v2_max_stg3_p75', 'x_hls_ndvi_v2_max_stg3_p90', 'x_hls_ndvi_v2_max_stg3_p95',
        'x_hls_ndvi_v2_max_stg4_max', 'x_hls_ndvi_v2_max_stg4_p75', 'x_hls_ndvi_v2_max_stg4_p90', 'x_hls_ndvi_v2_max_stg4_p95'
    ],
    # Min
    [
        'x_hls_ndvi_v2_min_stg1_min', 'x_hls_ndvi_v2_min_stg1_p5', 'x_hls_ndvi_v2_min_stg1_p10', 'x_hls_ndvi_v2_min_stg1_p25',
        'x_hls_ndvi_v2_min_stg2_min', 'x_hls_ndvi_v2_min_stg2_p5', 'x_hls_ndvi_v2_min_stg2_p10', 'x_hls_ndvi_v2_min_stg2_p25',
        'x_hls_ndvi_v2_min_stg3_min', 'x_hls_ndvi_v2_min_stg3_p5', 'x_hls_ndvi_v2_min_stg3_p10', 'x_hls_ndvi_v2_min_stg3_p25',
        'x_hls_ndvi_v2_min_stg4_min', 'x_hls_ndvi_v2_min_stg4_p5', 'x_hls_ndvi_v2_min_stg4_p10', 'x_hls_ndvi_v2_min_stg4_p25',
    ],
    # Max pctl
    [
        'x_hls_ndvi_v2_pctl_max_stg1_max', 'x_hls_ndvi_v2_pctl_max_stg1_p75', 'x_hls_ndvi_v2_pctl_max_stg1_p90', 'x_hls_ndvi_v2_pctl_max_stg1_p95',
        'x_hls_ndvi_v2_pctl_max_stg2_max', 'x_hls_ndvi_v2_pctl_max_stg2_p75', 'x_hls_ndvi_v2_pctl_max_stg2_p90', 'x_hls_ndvi_v2_pctl_max_stg2_p95',
        'x_hls_ndvi_v2_pctl_max_stg3_max', 'x_hls_ndvi_v2_pctl_max_stg3_p75', 'x_hls_ndvi_v2_pctl_max_stg3_p90', 'x_hls_ndvi_v2_pctl_max_stg3_p95',
        'x_hls_ndvi_v2_pctl_max_stg4_max', 'x_hls_ndvi_v2_pctl_max_stg4_p75', 'x_hls_ndvi_v2_pctl_max_stg4_p90', 'x_hls_ndvi_v2_pctl_max_stg4_p95',
    ],
    # Min pctl
    [
        'x_hls_ndvi_v2_pctl_min_stg1_min', 'x_hls_ndvi_v2_pctl_min_stg1_p5', 'x_hls_ndvi_v2_pctl_min_stg1_p10', 'x_hls_ndvi_v2_pctl_min_stg1_p25',
        'x_hls_ndvi_v2_pctl_min_stg2_min', 'x_hls_ndvi_v2_pctl_min_stg2_p5', 'x_hls_ndvi_v2_pctl_min_stg2_p10', 'x_hls_ndvi_v2_pctl_min_stg2_p25',
        'x_hls_ndvi_v2_pctl_min_stg3_min', 'x_hls_ndvi_v2_pctl_min_stg3_p5', 'x_hls_ndvi_v2_pctl_min_stg3_p10', 'x_hls_ndvi_v2_pctl_min_stg3_p25',
        'x_hls_ndvi_v2_pctl_min_stg4_min', 'x_hls_ndvi_v2_pctl_min_stg4_p5', 'x_hls_ndvi_v2_pctl_min_stg4_p10', 'x_hls_ndvi_v2_pctl_min_stg4_p25',
    ]
]

figure_xlabels = [
    "whssn_max", "whssn_min", "whssn_pctl_max", "whssn_pctl_min",
    "stg2to3_max", "stg2to3_min", "stg2to3_pctl_max", "stg2to3_pctl_min",
    "stg1-4_max", "stg1-4_min", "stg1-4_pctl_max", "stg1-4_pctl_min",
    
]
figure_title = "SharpDrop(Min) VS Background-BackScatter(Max)"
folder_name = "1.1.SharpDrop_VS_Background-BackScatter"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)

# Main features
features_main_hls = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main_hls)
#%%
list_feature_combinations = [
    ['x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_80_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_85_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_90_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_95_strict_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_80_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_85_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_90_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_95_relax_whssn_max', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_whssn_p75', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_whssn_p90', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_whssn_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_80_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_80_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_85_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_85_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_90_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_90_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_95_strict_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_95_strict_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_80_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_80_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_85_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_85_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_90_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_90_relax_stg2to3_p95'],
    ['x_hls_ndvi_v2_cnsct_period_under_95_relax_stg2to3_max', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_stg2to3_p75', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_stg2to3_p90', 'x_hls_ndvi_v2_cnsct_period_under_95_relax_stg2to3_p95'],
]




#%%
[column for column in df_tambon.columns.tolist() if ('x_hls_ndvi' in column)]



















