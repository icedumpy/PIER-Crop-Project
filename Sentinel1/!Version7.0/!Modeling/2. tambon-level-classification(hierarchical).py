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
def p5(x):
    return x.quantile(0.05)
def p10(x):
    return x.quantile(0.10)
def p25(x):
    return x.quantile(0.25)
def p75(x):
    return x.quantile(0.75)
def p90(x):
    return x.quantile(0.90)
def p95(x):
    return x.quantile(0.95)

def extract_and_combine_ranks(list_feature_combinations):
    return ["_".join(map(lambda val: val.split("_")[-1], features)) for features in list_feature_combinations]

def show_dist(df_tambon, column):
    plt.figure()
    plt.title(column)
    sns.histplot(
        data=df_tambon, x=column, 
        hue="y", common_norm=False, stat="probability"
    )

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
#%%
dict_agg_features = {
    "y":"max",
    # Sentinel-1
    "x_s1_bc_drop_min":["min", p5, p10, p25],
    "x_s1_bc_bc(t)_min":["min", p5, p10, p25],
    "x_s1_bc_drop+bc_min":["min", p5, p10, p25],
    "x_s1_bc_background_bc_minus_bc_t_max":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_1-6_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_7-12_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_13-18_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_19+_days_strict":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_1-6_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_7-12_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_13-18_days_relax":["max", p75, p90, p95],
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_19+_days_relax":["max", p75, p90, p95],
    # GISTDA Flood
    # "x_gistda_flood_ratio_0":["max", p75, p90, p95], #?
    "x_gistda_flood_ratio_1-5":["max", p75, p90, p95],
    "x_gistda_flood_ratio_6-10":["max", p75, p90, p95],
    "x_gistda_flood_ratio_11-15":["max", p75, p90, p95],
    "x_gistda_flood_ratio_15+":["max", p75, p90, p95],
    # "x_gistda_flood_ratio_relax_0":["max", p75, p90, p95],
    "x_gistda_flood_ratio_relax_1-5":["max", p75, p90, p95],
    "x_gistda_flood_ratio_relax_6-10":["max", p75, p90, p95],
    "x_gistda_flood_ratio_relax_11-15":["max", p75, p90, p95],
    "x_gistda_flood_ratio_relax_15+":["max", p75, p90, p95],
    # GSMap Rainfall
    "x_gsmap_rain_ph1_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph2_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph3_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph4_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph4a_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph4b_CR":["max", p75, p90, p95],
    "x_gsmap_rain_wh_ssn_CR":["max", p75, p90, p95],
    "x_gsmap_rain_0_105_CR":["max", p75, p90, p95],
    "x_gsmap_rain_106_120_CR":["max", p75, p90, p95],
    "x_gsmap_rain_ph1_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph2_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph3_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph4_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph4a_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph4b_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_wh_ssn_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_0_105_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_106_120_CWD":["max", p75, p90, p95],
    "x_gsmap_rain_ph1_ME":["max", p75, p90, p95],
    "x_gsmap_rain_ph2_ME":["max", p75, p90, p95],
    "x_gsmap_rain_ph3_ME":["max", p75, p90, p95],
    "x_gsmap_rain_ph4_ME":["max", p75, p90, p95],
    "x_gsmap_rain_ph4a_ME":["max", p75, p90, p95],
    "x_gsmap_rain_ph4b_ME":["max", p75, p90, p95],
    "x_gsmap_rain_wh_ssn_ME":["max", p75, p90, p95],
    "x_gsmap_rain_0_105_ME":["max", p75, p90, p95],
    "x_gsmap_rain_106_120_ME":["max", p75, p90, p95],
    # NECTEC Rain fall
    "x_nectec_rain_ph1_mean_rainfall":["max", p75, p90, p95],
    "x_nectec_rain_ph2_mean_rainfall":["max", p75, p90, p95],
    "x_nectec_rain_ph3_mean_rainfall":["max", p75, p90, p95],
    "x_nectec_rain_ph4_mean_rainfall":["max", p75, p90, p95],
    "x_nectec_rain_wh_ssn_mean_rainfall":["max", p75, p90, p95],
    # Soil Moisture
    "x_smap_soil_moist_max_sm":["max", p75, p90, p95],
    "x_smap_soil_moist_pctl_max_sm":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_80_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_80_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_85_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_85_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_90_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_90_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_95_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_above_95_relax":["max", p75, p90, p95],
    # HLS NDVI
    "x_hls_ndvi_v2_max_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_whssn":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_whssn":["min", p5, p10, p25],
    "x_hls_ndvi_v2_max_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_stg2to3":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_stg2to3":["min", p5, p10, p25],
    "x_hls_ndvi_v2_max_stg1":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_stg1":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_stg1":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_stg1":["min", p5, p10, p25],
    "x_hls_ndvi_v2_max_stg2":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_stg2":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_stg2":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_stg2":["min", p5, p10, p25],
    "x_hls_ndvi_v2_max_stg3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_stg3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_stg3":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_stg3":["min", p5, p10, p25],
    "x_hls_ndvi_v2_max_stg4":["max", p75, p90, p95],
    "x_hls_ndvi_v2_pctl_max_stg4":["max", p75, p90, p95],
    "x_hls_ndvi_v2_min_stg4":["min", p5, p10, p25],
    "x_hls_ndvi_v2_pctl_min_stg4":["min", p5, p10, p25],
    "x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_80_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_80_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_85_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_85_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_90_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_90_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_95_strict_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_95_relax_whssn":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_80_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_80_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_85_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_85_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_90_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_90_relax_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_95_strict_stg2to3":["max", p75, p90, p95],
    "x_hls_ndvi_v2_cnsct_period_above_95_relax_stg2to3":["max", p75, p90, p95],
    # MODIS NDVI
    "x_modis_ndvi_max_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_min_whssn":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_whssn":["min", p5, p10, p25],
    "x_modis_ndvi_max_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_min_stg2to3":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_stg2to3":["min", p5, p10, p25],
    "x_modis_ndvi_max_stg1":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_stg1":["max", p75, p90, p95],
    "x_modis_ndvi_min_stg1":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_stg1":["min", p5, p10, p25],
    "x_modis_ndvi_max_stg2":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_stg2":["max", p75, p90, p95],
    "x_modis_ndvi_min_stg2":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_stg2":["min", p5, p10, p25],
    "x_modis_ndvi_max_stg3":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_stg3":["max", p75, p90, p95],
    "x_modis_ndvi_min_stg3":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_stg3":["min", p5, p10, p25],
    "x_modis_ndvi_max_stg4":["max", p75, p90, p95],
    "x_modis_ndvi_pctl_max_stg4":["max", p75, p90, p95],
    "x_modis_ndvi_min_stg4":["min", p5, p10, p25],
    "x_modis_ndvi_pctl_min_stg4":["min", p5, p10, p25],
    "x_modis_ndvi_cnsct_period_under_5_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_5_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_10_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_10_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_15_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_15_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_20_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_20_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_80_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_80_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_85_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_85_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_90_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_90_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_95_strict_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_95_relax_whssn":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_5_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_5_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_10_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_10_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_15_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_15_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_20_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_under_20_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_80_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_80_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_85_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_85_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_90_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_90_relax_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_95_strict_stg2to3":["max", p75, p90, p95],
    "x_modis_ndvi_cnsct_period_above_95_relax_stg2to3":["max", p75, p90, p95]
}
#%%
# Save folder
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211220"
path_df_tambon = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_tambon.parquet"

# Load train test dataframe
df_list_train = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_train.csv").iloc[:, 1]
df_list_test  = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_test.csv").iloc[:, 1]

# Load tambon level features (if available)
if os.path.exists(path_df_tambon):
    df_tambon = pd.read_parquet(path_df_tambon)
# Load plot level then agg
else:
    # Load features dataframe
    df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.parquet")
    df = df[df["y"].isin([0, 3, 4])]
    df.loc[df["y"] != 0, "y"] = 1
    
    # Create tambon label features (Agg)
    df_tambon = df.groupby(["final_plant_year", "tambon_pcode"]).agg(dict_agg_features)
    df_tambon.columns = [df_tambon.columns.map("_".join)][0]
    df_tambon = df_tambon[~df_tambon.iloc[:, 1:].isna().any(axis=1)]
    df_tambon = df_tambon.reset_index()
    df_tambon = df_tambon.rename(columns={"y_max":"y"})
    df_tambon.to_parquet(path_df_tambon)
    del df
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
# =============================================================================
# 1.1.Sharp drop OR Backgroud-BC
# =============================================================================
# Defind parameters
list_feature_combinations = [
    ['x_s1_bc_drop_min_min'],
    ['x_s1_bc_drop_min_p5'],
    ['x_s1_bc_drop_min_p10'],
    ['x_s1_bc_drop_min_p25'],
    ['x_s1_bc_background_bc_minus_bc_t_max_max'],
    ['x_s1_bc_background_bc_minus_bc_t_max_p75'],
    ['x_s1_bc_background_bc_minus_bc_t_max_p90'],
    ['x_s1_bc_background_bc_minus_bc_t_max_p95'],
    ['x_s1_bc_drop_min_min', 'x_s1_bc_drop_min_p5', 
     'x_s1_bc_drop_min_p10', 'x_s1_bc_drop_min_p25'
    ],
    ['x_s1_bc_background_bc_minus_bc_t_max_p75', 'x_s1_bc_background_bc_minus_bc_t_max_p95',
     'x_s1_bc_background_bc_minus_bc_t_max_p95', 'x_s1_bc_background_bc_minus_bc_t_max_p95'
    ]
]
figure_xlabels = [
    "drop_min", "drop_p5", "drop_p10", "drop_p25",
    "BG-BS_max", "BG-BS_p75", "BG-BS_p90", "BG-BS_p95",
    "drop(All)", "BG-BS(All)"
]
figure_title = "SharpDrop(Min) VS Background-BackScatter(Max)"
folder_name = "1.1.SharpDrop_VS_Background-BackScatter"

# RUNNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

features_drop = df_report.loc[criteria].idxmax()[1].split("&")
print(features_drop)
#%%
# =============================================================================
# 1.2.Drop (Which combination?)
# =============================================================================
list_feature_combinations = get_combinations(features_drop)
figure_xlabels = ["_".join(map(lambda val: val.split("_")[-1], features)) for features in list_feature_combinations]
figure_title = "Drop (Combination)"
folder_name = "1.2.Drop_combination"

# RUNNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

features_drop = df_report.loc[criteria].idxmax()[1].split("&")
print(features_drop)
#%% 
# =============================================================================
# 2. Backscatter Level: Min (which combination?)
# =============================================================================
list_feature_combinations = get_combinations([
    "x_s1_bc_bc(t)_min_min",
    "x_s1_bc_bc(t)_min_p5",
    "x_s1_bc_bc(t)_min_p10",
    "x_s1_bc_bc(t)_min_p25",
])
figure_xlabels = ["_".join(map(lambda val: val.split("_")[-1], features)) for features in list_feature_combinations]
figure_title = "BackScatter (Min)"
folder_name = "2.BackScatter"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

features_bs_level = df_report.loc[criteria].idxmax()[1].split("&")
print(features_bs_level)
#%%
# =============================================================================
# 3. Backscatter Level+Drop: Min (which combination?)
# =============================================================================
list_feature_combinations = get_combinations([
    "x_s1_bc_drop+bc_min_min",
    "x_s1_bc_drop+bc_min_p5",
    "x_s1_bc_drop+bc_min_p10",
    "x_s1_bc_drop+bc_min_min",
])

figure_xlabels = ["_".join(map(lambda val: val.split("_")[-1], features)) for features in list_feature_combinations]
figure_title = "BackScatter+Drop (Min)"
folder_name = "3.BackScatter_Plus_Drop"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

features_bs_plus_drop_level = df_report.loc[criteria].idxmax()[1].split("&")
print(features_bs_plus_drop_level)
#%%
# =============================================================================
# 4. Drop OR Backscatter level OR Drop|Backscatter Or Drop+Backscatter
# =============================================================================
list_feature_combinations = [
    features_drop,
    features_bs_level,
    features_bs_plus_drop_level,
    features_drop+features_bs_level
]
figure_xlabels = ["Only drop", "Only BS", "drop+BS", "drop & BS"]
figure_title = "Drop VS BackScatter VS Drop+BackScatter VS Drop & Backscatter"
folder_name = "4.Drop_and_BackScatter_combination"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features!!
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# 5.1 Sentinel-1 Intensity (Select Threshold) 
# =============================================================================
#%%
list_feature_combinations = []
for threshold in [-12, -13, -14, -15, -16, -17, -18]:
   list_feature_combinations.append(features_main+[column for column in df_tambon.columns.tolist() if f"backscatter_under({threshold})" in column])
figure_xlabels = [f"BS_Under({threshold})" for threshold in [-12, -13, -14, -15, -16, -17, -18]]
figure_title = "Sentinel-1 Intensity (Threshold)"
folder_name = "5.1.Sentinel-1 Intensity (Threshold)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
#%%
# =============================================================================
# 5.2 After getting Threshold -> Which rank?
# =============================================================================
list_feature_combinations = []
list_rank_combinations = get_combinations(["max", "p75", "p90", "p95"])
threshold = int(df_report.loc[criteria].idxmax()[1].split("&")[-1].split("under(")[1][:3]) # Get threshold from string
for rank in list_rank_combinations:
    list_feature_combinations.append([column for column in df_tambon.columns.tolist() if ((f"backscatter_under({threshold})" in column)) and (column[-3:] in rank)])
figure_xlabels = ["_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "Sentinel-1 Intensity (Rank)"
folder_name = "5.2.Sentinel-1 Intensity (Rank)"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# 6.1 GISTDA Flood (Normal Or Relax)
# =============================================================================
list_feature_combinations = [
    features_main+[column for column in df_tambon.columns.tolist() if ("x_gistda_flood_ratio" in column) and (not "relax" in column)],
    features_main+[column for column in df_tambon.columns.tolist() if ("x_gistda_flood_ratio" in column) and ("relax" in column)],
]

figure_xlabels = ["GISTDA (strict)", "GISTDA (relax)"]
figure_title = "GISTDA Strict VS Relax"
folder_name = "6.1.GISTDA Strict VS Relax"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)
#%%
# =============================================================================
# 6.2 GISTDA Flood -> Which rank?
# =============================================================================
list_feature_combinations = []
list_rank_combinations = get_combinations(["max", "p75", "p90", "p95"])
strict_or_relax = "relax" if "Relax" in df_report.loc[criteria].idxmax()[0] else "strict"
for rank in list_rank_combinations:
    if strict_or_relax == "relax":
        list_feature_combinations.append([column for column in df_tambon.columns.tolist() if ("x_gistda_flood_ratio" in column) and ("relax" in column) and (column[-3:] in rank)])
    else:
        list_feature_combinations.append([column for column in df_tambon.columns.tolist() if ("x_gistda_flood_ratio" in column) and (not "relax" in column) and (column[-3:] in rank)])

figure_xlabels = ["_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "GISTDA (Rank)"
folder_name = "6.2.GISTDA (Rank)"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# 7.1 Rainfall GSMap (CR OR CWD OR ME)
# =============================================================================
list_feature_combinations = [
    features_main+[column for column in df_tambon.columns.tolist() if "x_gsmap_rain_wh_ssn_CR" in column],
    features_main+[column for column in df_tambon.columns.tolist() if "x_gsmap_rain_wh_ssn_CWD" in column],
    features_main+[column for column in df_tambon.columns.tolist() if "x_gsmap_rain_wh_ssn_ME" in column],
    features_main+[column for column in df_tambon.columns.tolist() if re.match(r"x_gsmap_rain_ph[0-9]+_CR", column)],
    features_main+[column for column in df_tambon.columns.tolist() if re.match(r"x_gsmap_rain_ph[0-9]+_CWD", column)],
    features_main+[column for column in df_tambon.columns.tolist() if re.match(r"x_gsmap_rain_ph[0-9]+_ME", column)]
]

figure_xlabels = ["CR Whole season", "CWD Whole season", "ME Whole season", "CR Growth stage", "CWD Growth stage", "ME Whole season"]
figure_title = "GSMap (Whole season VS Growth Stage)"
folder_name = "7.1.GSMap (Whole season VS Growth Stage)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)
#%%
# =============================================================================
# 7.2 Rainfall GSMap (Which rank?)
# =============================================================================
list_feature_combinations = []
features_temp = [feature for feature in df_report.loc[criteria].idxmax()[1].split("&") if not feature in features_main]
list_rank_combinations = get_combinations(["max", "p75", "p90", "p95"])
for rank in list_rank_combinations:
    list_feature_combinations.append([feature for feature in features_temp if feature[-3:] in rank])

figure_xlabels = ["_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "GSMap (Rank)"
folder_name = "7.2.GSMap (Rank)"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# 8.Soil moisture (Level)
# =============================================================================
list_feature_combinations = get_combinations(['x_smap_soil_moist_max_sm_max', 'x_smap_soil_moist_max_sm_p75', 'x_smap_soil_moist_max_sm_p90', 'x_smap_soil_moist_max_sm_p95']) + get_combinations(['x_smap_soil_moist_pctl_max_sm_max', 'x_smap_soil_moist_pctl_max_sm_p75', 'x_smap_soil_moist_pctl_max_sm_p90', 'x_smap_soil_moist_pctl_max_sm_p95'])

figure_xlabels = ["pctl_"+"_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) if "pctl" in features[0] else "_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "Soil moisture"
folder_name = "8.1.Soil moisture"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# 9.1.Soil moisture (Intensity)
# =============================================================================
list_feature_combinations = []
figure_xlabels = []
for pctl in [80, 85, 90, 95]:
    for strict_or_relax in ["strict", "relax"]:
        list_feature_combinations.append(features_main+[column for column in df_tambon.columns.tolist() if ('x_smap_soil_moist_v2_cnsct_period' in column) and (f"_{pctl}_" in column) and (strict_or_relax in column)])
        figure_xlabels.append(f"{pctl}_{strict_or_relax}")

figure_title = "Soil moisture (Intensity)"
folder_name = "9.1.Soil moisture (Intensity)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)
#%%
# =============================================================================
# 9.2.Soil moisture (Which rank)
# =============================================================================
list_feature_combinations = get_combinations([column for column in df_tambon.columns.tolist() if ('x_smap_soil_moist_v2_cnsct_period' in column) and (f"_{pctl}_" in column) and (strict_or_relax in column)])
pctl, strict_or_relax = df_report.loc[criteria].idxmax()[0].split("_")

figure_xlabels = ["pctl_"+"_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) if "pctl" in features[0] else "_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "Soil moisture (Rank)"
folder_name = "9.2.Soil moisture (Rank)"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main)
#%%
# =============================================================================
# Another main topic NDVI (Best of MODIS vs Best of HLS)
# =============================================================================
#%%
# =============================================================================
# 10.1.HLS NDVI (Level)
# =============================================================================
list_feature_combinations = get_combinations(['x_hls_ndvi_v2_min_whssn_min', 'x_hls_ndvi_v2_min_whssn_p5', 'x_hls_ndvi_v2_min_whssn_p10', 'x_hls_ndvi_v2_min_whssn_p25']) + get_combinations(['x_hls_ndvi_v2_pctl_min_whssn_min', 'x_hls_ndvi_v2_pctl_min_whssn_p5', 'x_hls_ndvi_v2_pctl_min_whssn_p10', 'x_hls_ndvi_v2_pctl_min_whssn_p25'])
figure_xlabels = ["pctl_"+"_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) if "pctl" in features[0] else "_".join(np.unique(list(map(lambda val: val.split("_")[-1], features))).tolist()) for features in list_feature_combinations]
figure_title = "HLS NDVI"
folder_name = "10.1.HLS NDVI (Level)"

# Add Main features
list_feature_combinations = [features_main+features for features in list_feature_combinations]

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_main_hls = df_report.loc[criteria].idxmax()[1].split("&")
print(features_main_hls)
#%%
# =============================================================================
# 10.2.HLS NDVI (Intensity)
# =============================================================================
#%%
list_feature_combinations = []
figure_xlabels = []

for pctl in [5, 10, 15, 20]:
    for strict_or_relax in ["strict", "relax"]:
        list_feature_combinations.append(features_main+[column for column in df_tambon.columns.tolist() if ('x_hls_ndvi_v2_cnsct_period' in column) and (f"_{pctl}_" in column) and (strict_or_relax in column)])
        figure_xlabels.append(f"{pctl}_{strict_or_relax}")


# [column for column in df_tambon.columns.tolist() if ('x_hls_ndvi_v2_cnsct_period' in column) and (f"_{pctl}_" in column)]

#%%
#%%
# x_train, y_train = df_tambon_train[features_main].values, df_tambon_train["y"].values
# x_test,  y_test  = df_tambon_test[features_main].values, df_tambon_test["y"].values

# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, criterion="gini", n_jobs=-1))
# ])
# pipeline.fit(x_train, y_train)

# # Get predicted
# y_test_pred  = pipeline.predict(x_test)

# # Calculate score
# f1 = f1_score(y_test, y_test_pred)

# # Confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_test_pred)
# #%%
# plt.close("all")
# fig, ax = plt.subplots()
# plot_roc_curve(pipeline, x_train, y_train, label="Train", color="g-", ax=ax)
# plot_roc_curve(pipeline, x_test, y_test, label="Test", color="r--", ax=ax)
# ax.legend()
# ax.set_title(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
# fig.savefig(os.path.join(root_save, "ROC.png"), bbox_inches="tight")

# plt.close("all")
# fig, ax = plt.subplots()
# plot_confusion_matrix(pipeline, x_test, y_test, ax=ax)
# ax.set_title("Confusion Matrix (Test)")
# fig.savefig(os.path.join(root_save, "confusion_matrix.png"), bbox_inches="tight")