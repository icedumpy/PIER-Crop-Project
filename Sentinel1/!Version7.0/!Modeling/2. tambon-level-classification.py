import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score
#%%
def plot_report(list_dict_report, label=None):
    df_report = make_report(list_dict_report)
    fig, ax = plt.subplots(figsize=(16, 9))
    df_report.plot.bar(rot=0, ax=ax)
    ax.set_ylim(0, 1)
    return fig, ax, df_report
    
def make_report(list_dict_report):
    list_df = []
    for dict_report in list_dict_report:
        list_df.append(
            pd.DataFrame([
                dict_report["f1_binary"],
                dict_report["f1_macro"],
                dict_report["f1_weighted"],
            ], columns = ["&".join(dict_report["features"])], index=["f1_binary", "f1_macro", "f1_weighted"])
        )
    df_report = pd.concat(list_df, axis=1)
    return df_report

def show_dist(df_tambon, column):
    plt.figure()
    plt.title(column)
    sns.histplot(
        data=df_tambon, x=column, 
        hue="y", common_norm=False, stat="probability"
    )

def run_model(x_train, y_train, x_test, y_test, features):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, criterion="gini", class_weight="balanced_subsample", n_jobs=-1))
    ])
    pipeline.fit(x_train, y_train)
    
    # Get predicted
    y_test_pred  = pipeline.predict(x_test)
    
    # Calculate score
    f1_binary = f1_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average="macro")
    f1_weighted = f1_score(y_test, y_test_pred, average="weighted")
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    
    dict_report = {
        "features" : features,
        "f1_binary" : f1_binary,
        "f1_macro" : f1_macro,
        "f1_weighted" : f1_weighted,
        "confustion_matrix": cnf_matrix
    }
    return dict_report
#%%
dict_agg_features = {
    "y":"max",
    # Sentinel-1
    # "x_s1_bc_drop_min":"p,5, 10, 25",
    "x_s1_bc_drop_min":"min",
    "x_s1_bc_bc(t)_min":"min",
    "x_s1_bc_drop+bc_min":"min",
    "x_s1_bc_background_bc_minus_bc_t_max":"max",
    # "x_s1_bc_background_bc_minus_bc_t_max":"p75, 90, 95",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-12)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-13)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-14)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-15)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-16)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-17)_19+_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_1-6_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_7-12_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_13-18_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_19+_days_strict":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_1-6_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_7-12_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_13-18_days_relax":"max",
    "x_s1_bc_v2_pct_of_plot_with_backscatter_under(-18)_19+_days_relax":"max",
    # GISTDA Flood
    # "x_gistda_flood_ratio_0":"max", # Check ?
    "x_gistda_flood_ratio_1-5":"max",
    "x_gistda_flood_ratio_6-10":"max",
    "x_gistda_flood_ratio_11-15":"max",
    "x_gistda_flood_ratio_15+":"max",
    # "x_gistda_flood_ratio_relax_0":"max",
    "x_gistda_flood_ratio_relax_1-5":"max",
    "x_gistda_flood_ratio_relax_6-10":"max",
    "x_gistda_flood_ratio_relax_11-15":"max",
    "x_gistda_flood_ratio_relax_15+":"max",
    # Rainfall
    "x_gsmap_rain_ph1_CR":"max",
    "x_gsmap_rain_ph2_CR":"max",
    "x_gsmap_rain_ph3_CR":"max",
    "x_gsmap_rain_ph4_CR":"max",
    "x_gsmap_rain_ph4a_CR":"max",
    "x_gsmap_rain_ph4b_CR":"max",
    "x_gsmap_rain_wh_ssn_CR":"max",
    "x_gsmap_rain_0_105_CR":"max",
    "x_gsmap_rain_106_120_CR":"max",
    "x_gsmap_rain_ph1_CWD":"max",
    "x_gsmap_rain_ph2_CWD":"max",
    "x_gsmap_rain_ph3_CWD":"max",
    "x_gsmap_rain_ph4_CWD":"max",
    "x_gsmap_rain_ph4a_CWD":"max",
    "x_gsmap_rain_ph4b_CWD":"max",
    "x_gsmap_rain_wh_ssn_CWD":"max",
    "x_gsmap_rain_0_105_CWD":"max",
    "x_gsmap_rain_106_120_CWD":"max",
    # Soil Moisture
    "x_smap_soil_moist_max_sm":"max",
    "x_smap_soil_moist_pctl_max_sm":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_80_strict":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_80_relax":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_85_strict":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_85_relax":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_90_strict":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_90_relax":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_95_strict":"max",
    "x_smap_soil_moist_v2_cnsct_period_above_95_relax":"max",
    # NDVI
    "x_hls_ndvi_v2_min_whssn":"min",
    "x_hls_ndvi_v2_pctl_min_whssn":"min",
    "x_hls_ndvi_v2_min_stg2to3":"min",
    "x_hls_ndvi_v2_pctl_min_stg2to3":"min",
    "x_hls_ndvi_v2_min_stg1":"min",
    "x_hls_ndvi_v2_pctl_min_stg1":"min",
    "x_hls_ndvi_v2_min_stg2":"min",
    "x_hls_ndvi_v2_pctl_min_stg2":"min",
    "x_hls_ndvi_v2_min_stg3":"min",
    "x_hls_ndvi_v2_pctl_min_stg3":"min",
    "x_hls_ndvi_v2_min_stg4":"min",
    "x_hls_ndvi_v2_pctl_min_stg4":"min",
    "x_hls_ndvi_v2_cnsct_period_under_5_strict_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_5_relax_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_10_strict_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_10_relax_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_15_strict_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_15_relax_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_20_strict_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_20_relax_whssn":"max",
    "x_hls_ndvi_v2_cnsct_period_under_5_strict_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_5_relax_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_10_strict_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_10_relax_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_15_strict_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_15_relax_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_20_strict_stg2to3":"max",
    "x_hls_ndvi_v2_cnsct_period_under_20_relax_stg2to3":"max"
}
#%%
# Save folder
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20211220"

# Load features dataframe
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.parquet")
df = df[df["y"].isin([0, 3, 4])]
df.loc[df["y"] != 0, "y"] = 1

# Load train test dataframe
df_list_train = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_train.csv").iloc[:, 1]
df_list_test  = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_test.csv").iloc[:, 1]

# Create tambon label features
df_tambon = df.groupby(["final_plant_year", "tambon_pcode"]).agg(dict_agg_features)
df_tambon = df_tambon[~df_tambon.iloc[:, 1:].isna().any(axis=1)]
df_tambon = df_tambon.reset_index()
del df
# Visualize features distribution
# columns = list(dict_agg_features.keys())[1:]
# plt.close("all")
# for column in columns:
#     if "hls" in column:
#         plt.figure()
#         plt.title(column)
#         sns.histplot(
#             data=df_tambon, x=column, 
#             hue="y", common_norm=False, stat="probability"
#         )
#%% Separate train test
df_tambon_train = df_tambon[df_tambon["tambon_pcode"].isin(df_list_train)]
df_tambon_test  = df_tambon[df_tambon["tambon_pcode"].isin(df_list_test)]
#%%
plt.close("all")
criteria = "f1_binary"
dict_master_report = dict()
dict_master_report["features"] = []
dict_master_report["f1"] = []

# =============================================================================
# Level 1: Sharp drop (pure sharp drop vs background minus backscatter)
# "x_s1_bc_drop_min" vs "x_s1_bc_background_bc_minus_bc_t_max"
# =============================================================================
features_level_1_1 = ["x_s1_bc_drop_min"]
features_level_1_2 = ["x_s1_bc_background_bc_minus_bc_t_max"]

x_train, y_train = df_tambon_train[features_level_1_1].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_1_1], df_tambon_test["y"]
dict_level_1_1 = run_model(x_train, y_train, x_test, y_test, features_level_1_1)

x_train, y_train = df_tambon_train[features_level_1_2].values.reshape(-1, 1), df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_1_2].values.reshape(-1, 1), df_tambon_test["y"]
dict_level_1_2 = run_model(x_train, y_train, x_test, y_test, features_level_1_2)

# Make report and plot
list_dict_report = [dict_level_1_1, dict_level_1_2]
fig, ax, df_report = plot_report(list_dict_report)
fig.savefig(os.path.join(root_save, "1.Level1-inner.png"), bbox_inches="tight")

# Select winner from Level 1
features_level_1 = [df_report.loc[criteria].idxmax()]
print(f"Level 1 winner: {features_level_1}")
dict_master_report["features"].append(features_level_1)
dict_master_report["f1"].append(df_report.loc[criteria, features_level_1].values[0])
#%%
# =============================================================================
# Level 2 & 3: Backscatter level vs Backscatter level + drop 
# =============================================================================
features_level_2_1 = ["x_s1_bc_bc(t)_min"]
features_level_2_2 = ["x_s1_bc_drop+bc_min"]

x_train, y_train = df_tambon_train[features_level_2_1].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_2_1], df_tambon_test["y"]
dict_level_2_1 = run_model(x_train, y_train, x_test, y_test, features_level_2_1)

x_train, y_train = df_tambon_train[features_level_2_2].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_2_2], df_tambon_test["y"]
dict_level_2_2 = run_model(x_train, y_train, x_test, y_test, features_level_2_2)

# Make report and plot
list_dict_report = [dict_level_2_1, dict_level_2_2]
fig, ax, df_report = plot_report(list_dict_report)
fig.savefig(os.path.join(root_save, "2.Level2-inner.png"), bbox_inches="tight")

# Select winner from Level 2
features_level_2 = [df_report.loc[criteria].idxmax()]
print(f"Level 2 winner: {features_level_2}")
dict_master_report["features"].append(features_level_2)
dict_master_report["f1"].append(df_report.loc[criteria, features_level_2].values[0])
#%%
# =============================================================================
# Compare Level 1 only, Level 2 only, Level 1&2 
# =============================================================================
x_train, y_train = df_tambon_train[features_level_1].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_1], df_tambon_test["y"]
dict_level_1 = run_model(x_train, y_train, x_test, y_test, features_level_1)

x_train, y_train = df_tambon_train[features_level_2].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_2], df_tambon_test["y"]
dict_level_2 = run_model(x_train, y_train, x_test, y_test, features_level_2)

x_train, y_train = df_tambon_train[features_level_1+features_level_2].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_level_1+features_level_2], df_tambon_test["y"]
dict_level_1_2 = run_model(x_train, y_train, x_test, y_test, features_level_1+features_level_2)

# Make report and plot
list_dict_report = [dict_level_1, dict_level_2, dict_level_1_2]
fig, ax, df_report = plot_report(list_dict_report)
fig.savefig(os.path.join(root_save, "3.Level1&2_comparison.png"), bbox_inches="tight")

# Select winner from Level 1&2
features_main = df_report.loc[criteria].idxmax().split("&")
print(f"Level 1&2 winner: {features_main}")
dict_master_report["features"].append(features_main)
dict_master_report["f1"].append(df_report.loc[criteria, "&".join(features_main)])
#%%
# =============================================================================
# Intensity S1 ("Threshold")
# =============================================================================
list_dict_report = []
for threshold in [-12, -13, -14, -15,- 16,- 17,- 18]:
    features_intensity = [column for column in list(dict_agg_features.keys()) if f"x_s1_bc_v2_pct_of_plot_with_backscatter_under({threshold})" in column]
    features = features_main+features_intensity
    x_train, y_train = df_tambon_train[features].values, df_tambon_train["y"]
    x_test, y_test   = df_tambon_test[features], df_tambon_test["y"]
    list_dict_report.append(run_model(x_train, y_train, x_test, y_test, features))   
# Make report and plot
fig, ax, df_report = plot_report(list_dict_report)
ax.legend(labels=[-12, -13, -14, -15,- 16,- 17,- 18])
fig.savefig(os.path.join(root_save, "4.S1_intensity.png"), bbox_inches="tight")

# Select winner from threshold
features_main = df_report.loc[criteria].idxmax().split("&")
print(f"S1 Intensity winner: {features_main}")
dict_master_report["features"].append(features_main)
dict_master_report["f1"].append(df_report.loc[criteria, "&".join(features_main)])
#%%
# =============================================================================
# Intensity GISTDA ("Normal" or "Relax")
# =============================================================================
list_normal = ["x_gistda_flood_ratio_0", "x_gistda_flood_ratio_1-5", "x_gistda_flood_ratio_6-10", "x_gistda_flood_ratio_11-15", "x_gistda_flood_ratio_15+"]
list_relax = ["x_gistda_flood_ratio_relax_0", "x_gistda_flood_ratio_relax_1-5", "x_gistda_flood_ratio_relax_6-10", "x_gistda_flood_ratio_relax_11-15", "x_gistda_flood_ratio_relax_15+"]
list_dict_report = []
for features in [list_normal, list_relax]:
    features = features_main+features
    x_train, y_train = df_tambon_train[features].values, df_tambon_train["y"]
    x_test, y_test   = df_tambon_test[features], df_tambon_test["y"]
    list_dict_report.append(run_model(x_train, y_train, x_test, y_test, features))
# Make report and plot
fig, ax, df_report = plot_report(list_dict_report)
ax.legend(labels=["GISTDA_Normal", "GISTDA_Relax"])
fig.savefig(os.path.join(root_save, "5.GISTDA_intensity.png"), bbox_inches="tight")

# Select winner from Level 1&2
features_main = df_report.loc[criteria].idxmax().split("&")
print(f"GISTDA Intensity winner: {features_main}")
dict_master_report["features"].append(features_main)
dict_master_report["f1"].append(df_report.loc[criteria, "&".join(features_main)])
#%%
# =============================================================================
# Rainfall ??
# =============================================================================
# "x_gsmap_rain_ph1_CR"
# "x_gsmap_rain_ph2_CR"
# "x_gsmap_rain_ph3_CR"
# "x_gsmap_rain_ph4_CR"
# "x_gsmap_rain_ph4a_CR"
# "x_gsmap_rain_ph4b_CR"
# "x_gsmap_rain_wh_ssn_CR"
# "x_gsmap_rain_0_105_CR"
# "x_gsmap_rain_106_120_CR"
# "x_gsmap_rain_ph1_CWD"
# "x_gsmap_rain_ph2_CWD"
# "x_gsmap_rain_ph3_CWD"
# "x_gsmap_rain_ph4_CWD"
# "x_gsmap_rain_ph4a_CWD"
# "x_gsmap_rain_ph4b_CWD"
# "x_gsmap_rain_wh_ssn_CWD"
# "x_gsmap_rain_0_105_CWD"
# "x_gsmap_rain_106_120_CWD"
#%%
# =============================================================================
# Soil moisture (Level)
# =============================================================================
features_1 = features_main+["x_smap_soil_moist_max_sm"]
features_2 = features_main+["x_smap_soil_moist_pctl_max_sm"]

x_train, y_train = df_tambon_train[features_1].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_1], df_tambon_test["y"]
dict_level_1 = run_model(x_train, y_train, x_test, y_test, features_1)

x_train, y_train = df_tambon_train[features_2].values, df_tambon_train["y"]
x_test, y_test = df_tambon_test[features_2], df_tambon_test["y"]
dict_level_2 = run_model(x_train, y_train, x_test, y_test, features_2)

# Make report and plot
list_dict_report = [dict_level_1, dict_level_2]
fig, ax, df_report = plot_report(list_dict_report)
ax.legend(labels=["Max", "Percentile (Max)"])
fig.savefig(os.path.join(root_save, "7.SoilMoisture_level.png"), bbox_inches="tight")

# Select winner from Level 2
features_main = df_report.loc[criteria].idxmax().split("&")
print(f"Soil moisture level winner: {features}")
dict_master_report["features"].append(features_main)
dict_master_report["f1"].append(df_report.loc[criteria, "&".join(features_main)])
#%%


# =============================================================================
# Soil moisture (Intensity)
# =============================================================================

"x_smap_soil_moist_v2_cnsct_period_above_80_strict"
"x_smap_soil_moist_v2_cnsct_period_above_80_relax"
"x_smap_soil_moist_v2_cnsct_period_above_85_strict"
"x_smap_soil_moist_v2_cnsct_period_above_85_relax"
"x_smap_soil_moist_v2_cnsct_period_above_90_strict"
"x_smap_soil_moist_v2_cnsct_period_above_90_relax"
"x_smap_soil_moist_v2_cnsct_period_above_95_strict"
"x_smap_soil_moist_v2_cnsct_period_above_95_relax"

#%%
df_master_report = pd.DataFrame(dict_master_report)
df_master_report.to_csv(r"F:\CROP-PIER\CROP-WORK\Presentation\20211220\report.csv", index=False)



































