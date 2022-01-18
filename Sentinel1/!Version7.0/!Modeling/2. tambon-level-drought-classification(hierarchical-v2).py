import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import combinations
from imblearn.over_sampling import SMOTE 
from icedumpy.plot_tools import plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
def ratio(x):
    return (x == 1).sum()/len(x)

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

def add_rice_area(df, df_tambon):
    # Calculate tambon area (sq. wa)
    gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp")
    gdf["tambon_pcode"] = gdf["ADM3_PCODE"].str.slice(2,).astype("int32")
    gdf = gdf.loc[gdf["tambon_pcode"].isin(df["tambon_pcode"])]
    gdf = gdf.to_crs({"init":'epsg:32647'})
    gdf["tambon_area_in_wa"] = gdf.geometry.area/4.0
    
    # Calculate mean percentage rice area (each year)
    temp = df.groupby(["final_plant_year", "tambon_pcode"]).agg({"total_actual_plant_area_in_wa":"sum"})
    temp = temp.reset_index()
    temp = pd.merge(temp, gdf[["tambon_pcode", "tambon_area_in_wa"]], how="left", on="tambon_pcode")
    temp["x_percent_rice_area"] = temp["total_actual_plant_area_in_wa"]/temp["tambon_area_in_wa"]
    
    # Calculate mean percentage rice area
    temp = temp.groupby("tambon_pcode").agg({"x_percent_rice_area":"mean"})
    temp = temp.reset_index()

    # Finish
    df_tambon = pd.merge(df_tambon, temp, how="left", on="tambon_pcode")
    return df_tambon
#%%
dict_agg_features = {
    "y":"max",
    # Rice characteristics
    "x_rice_age_days":["mean"],
    "x_photo_sensitive_f":[ratio],
    "x_jasmine_rice_f":[ratio],
    "x_sticky_rice_f":[ratio],
    "x_plant_info_v2_irrigation_f":[ratio],  
    # DEM
    "x_dem_elevation":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_dem_gradient":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    # Sentinel-1
    "x_s1_bc_bc(t-2)_min":["min", p5, p10, p25],
    "x_s1_bc_bc(t-1)_min":["min", p5, p10, p25],
    "x_s1_bc_bc(t)_min":["min", p5, p10, p25],
    "x_s1_bc_bc(t+1)_min":["min", p5, p10, p25],
    "x_s1_bc_bc(t+2)_min":["min", p5, p10, p25],
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
    # GISTDA DRI
    "x_gistda_dri_max_dri":["max", p75, p90, p95],
    "x_gistda_dri_pctl_max_dri":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_80_strict":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_80_relax":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_85_strict":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_85_relax":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_90_strict":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_90_relax":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_95_strict":["max", p75, p90, p95],
    "x_gistda_dri_v2_cnsct_period_above_95_relax":["max", p75, p90, p95],
    # GSMap Rainfall
    "x_gsmap_rain_ph1_CR":["min", p5, p10, p25],
    "x_gsmap_rain_ph2_CR":["min", p5, p10, p25],
    "x_gsmap_rain_ph3_CR":["min", p5, p10, p25],
    "x_gsmap_rain_ph4_CR":["min", p5, p10, p25],
    "x_gsmap_rain_wh_ssn_CR":["min", p5, p10, p25],
    "x_gsmap_rain_ph1_MD":["max", p75, p90, p95],
    "x_gsmap_rain_ph2_MD":["max", p75, p90, p95],
    "x_gsmap_rain_ph3_MD":["max", p75, p90, p95],
    "x_gsmap_rain_ph4_MD":["max", p75, p90, p95],
    "x_gsmap_rain_wh_ssn_MD":["max", p75, p90, p95],
    "x_gsmap_rain_ph1_CDD":["max", p75, p90, p95],
    "x_gsmap_rain_ph2_CDD":["max", p75, p90, p95],
    "x_gsmap_rain_ph3_CDD":["max", p75, p90, p95],
    "x_gsmap_rain_ph4_CDD":["max", p75, p90, p95],
    "x_gsmap_rain_wh_ssn_CDD":["max", p75, p90, p95],
    # Soil Moisture
    "x_smap_soil_moist_min_sm":["min", p5, p10, p25],
    "x_smap_soil_moist_pctl_min_sm":["min", p5, p10, p25],
    "x_smap_soil_moist_v2_cnsct_period_under_5_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_5_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_10_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_10_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_15_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_15_relax":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_20_strict":["max", p75, p90, p95],
    "x_smap_soil_moist_v2_cnsct_period_under_20_relax":["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_strict_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_relax_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_strict_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_relax_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_strict_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_relax_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_strict_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_relax_stg1':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_strict_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_relax_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_strict_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_relax_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_strict_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_relax_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_strict_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_relax_stg2':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_strict_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_relax_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_strict_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_relax_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_strict_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_relax_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_strict_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_relax_stg3':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_strict_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_5_relax_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_strict_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_10_relax_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_strict_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_15_relax_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_strict_stg4':["max", p75, p90, p95],
    'x_smap_soil_moist_v3_cnsct_period_under_20_relax_stg4':["max", p75, p90, p95],
    # HLS NDVI
    "x_hls_ndvi_v2_min_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_med_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_max_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_min_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_med_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_max_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_min_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_med_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_max_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_min_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_med_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_max_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_min_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_med_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_max_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_min_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_med_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_max_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_min_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_med_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_max_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_min_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_med_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_max_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_min_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_med_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_max_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_min_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_med_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_pctl_max_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_80_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_80_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_85_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_85_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_90_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_90_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_95_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_hls_ndvi_v2_cnsct_period_above_95_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    # MODIS NDVI
    "x_modis_ndvi_min_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_med_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_max_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_min_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_med_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_max_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_min_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_med_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_max_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_min_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_med_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_max_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_min_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_med_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_max_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_min_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_med_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_max_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_min_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_med_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_max_stg1":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_min_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_med_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_max_stg2":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_min_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_med_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_max_stg3":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_min_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_med_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_pctl_max_stg4":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],    
    "x_modis_ndvi_cnsct_period_under_5_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_5_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_10_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_10_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_15_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_15_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_20_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_under_20_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_80_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_80_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_85_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_85_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_90_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_90_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_95_strict_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
    "x_modis_ndvi_cnsct_period_above_95_relax_whssn":["min", p5, p10, p25, "mean", "median", p75, p90, p95, "max"],
}
#%%
# Save folder
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211220\Drought"
path_df_tambon = r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_4a_NE3_tambon_drought.parquet"

# Load train test dataframe
df_list_train = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_train.csv").iloc[:, 1]
df_list_test  = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_test.csv").iloc[:, 1]

# Load tambon level features (if available)
if os.path.exists(path_df_tambon):
    df_tambon = pd.read_parquet(path_df_tambon)
# Load plot level then agg
else:
    # Load features dataframe
    df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_4a_NE3_compressed.parquet")
    df = df[df["y"].isin([0, 1, 2])]
    df.loc[df["y"] != 0, "y"] = 1
    
    # Create tambon label features (Agg)
    df_tambon = df.groupby(["final_plant_year", "tambon_pcode"]).agg(dict_agg_features)
    df_tambon.columns = [df_tambon.columns.map("_".join)][0]
    df_tambon = df_tambon[~df_tambon.iloc[:, 1:].isna().any(axis=1)]
    df_tambon = df_tambon.reset_index()
    df_tambon = df_tambon.rename(columns={"y_max":"y"})
    df_tambon = add_rice_area(df, df_tambon)
    df_tambon.to_parquet(path_df_tambon)
    del df
#%% Separate train test
df_tambon_train = df_tambon[df_tambon["tambon_pcode"].isin(df_list_train)]
df_tambon_test  = df_tambon[df_tambon["tambon_pcode"].isin(df_list_test)]

# Upsampling minority class (Only training data)
columns_training_feature = [column for column in df_tambon.columns if column.startswith("x_")]
df_tambon_train = pd.concat(SMOTE(sampling_strategy="minority", random_state=42).fit_resample(df_tambon_train[columns_training_feature], df_tambon_train["y"]), axis=1)
#%%
n_trials = 2
criteria = "f1"
list_report_main = []
#%%
# =============================================================================
# 0.Area & Plant characteristic (Control variables)
# =============================================================================
#%%
list_feature_combinations = [
    ['x_rice_age_days_mean', 'x_photo_sensitive_f_ratio', 'x_jasmine_rice_f_ratio',
     'x_sticky_rice_f_ratio', 'x_percent_rice_area', "x_plant_info_v2_irrigation_f_ratio",
     'x_dem_elevation_min', 'x_dem_elevation_median', 'x_dem_elevation_max',
     'x_dem_gradient_min', 'x_dem_gradient_median', 'x_dem_gradient_max']
]
figure_xlabels = [
   "Control variables"
]
figure_title = "Control Variables"
folder_name = "0.Control Variables"

# RUNNNN
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
# 1.HLS NDVI (Extreme)
# =============================================================================
list_feature_combinations = []
figure_xlabels = []
for stg in ["stg"]:
    for rank1 in ["min", "max"]:
        for rank2 in ["min", "p5", "p10", "p25", "mean", "median", "p75", "p90", "p95", "max"]:
            list_feature_combinations.append([column for column in df_tambon.columns.tolist() if  ("hls" in column) and (not "cnsct" in column) and (stg in column) and (f"v2_{rank1}" in column) and (column.split("_")[-1] == rank2)])
            figure_xlabels.append(f"{rank1}_{stg}_{rank2}")
figure_title = "HLS NDVI (Extreme)"
folder_name = "1.HLS NDVI (Extreme)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_hls_extreme = df_report.loc[criteria].idxmax()[1].split("&")
print(features_hls_extreme)

# Get top-n features
features_top_hls_extreme = df_report.loc[criteria].sort_values(ascending=False).iloc[:10].index.get_level_values(1).str.split("&").tolist()
figure_xlabels_top_hls_extreme = df_report.loc[criteria].sort_values(ascending=False).iloc[:10].index.get_level_values(0).str.split("&").tolist()
#%%
# =============================================================================
# 2. HLS NDVI (Intensity)
# =============================================================================
list_feature_combinations = []
figure_xlabels = []
for feature in [column for column in df_tambon.columns.tolist() if  ("x_hls_ndvi_v2_cnsct_period" in column)]:
    list_feature_combinations.append([feature])
    figure_xlabels.append(f"{''.join(feature.split('_')[-5:-3])}_{feature.split('_')[-3][0]}_{feature.split('_')[-1]}")
figure_title = "HLS NDVI (Intensity)"
folder_name = "2.HLS NDVI (Intensity)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_hls_intensity = df_report.loc[criteria].idxmax()[1].split("&")
print(features_hls_intensity)

# Get top-20 features
features_top_hls_intensity = df_report.loc[criteria].sort_values(ascending=False).iloc[:20].index.get_level_values(1).str.split("&").tolist()
figure_xlabels_top_hls_intensity = df_report.loc[criteria].sort_values(ascending=False).iloc[:20].index.get_level_values(0).str.split("&").tolist()
#%%
# =============================================================================
# 3. HLS NDVI (Extreme + Intensity)
# =============================================================================
list_feature_combinations = []
figure_xlabels = []
for extreme, extreme_xlabel in zip(features_top_hls_extreme, figure_xlabels_top_hls_extreme):
    for intensity, intensity_xlabel in zip(features_top_hls_intensity, figure_xlabels_top_hls_intensity):
        list_feature_combinations.append(extreme+intensity)
        figure_xlabels.append("&".join(extreme_xlabel+intensity_xlabel))
figure_title = "HLS NDVI (Extreme+Intensity)"
folder_name = "3.HLS NDVI (Extreme+Intensity)"

# RUNNN
df_report = main_features_comparison(
    df_tambon_train, df_tambon_test, list_feature_combinations, criteria, 
    folder_name=folder_name, figure_name="F1_comparison.png", report_name="Report.csv",
    figure_xlabels=figure_xlabels, figure_title=figure_title, n_trials=n_trials
)
list_report_main.append(df_report)

# Main features
features_hls_extreme_intensity = df_report.loc[criteria].idxmax()[1].split("&")
print(features_hls_extreme_intensity)
#%%














