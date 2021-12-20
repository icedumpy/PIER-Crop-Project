import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#%%
dict_agg_features = {
    "y":"max",
    # Sentinel-1
    "x_s1_bc_drop_min":"min",
    "x_s1_bc_bc(t)_min":"min",
    "x_s1_bc_drop+bc_min":"min",
    "x_s1_bc_background_bc_minus_bc_t_max":"max",
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
    "x_gistda_flood_ratio_0":"max",
    "x_gistda_flood_ratio_1-5":"max",
    "x_gistda_flood_ratio_6-10":"max",
    "x_gistda_flood_ratio_11-15":"max",
    "x_gistda_flood_ratio_15+":"max",
    "x_gistda_flood_ratio_relax_0":"max",
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
    # Don't think it will be useful.
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
# Load features dataframe
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\batch_3c\df_pierxda_batch_3c_NE1_compressed.parquet")
df = df[df["y"].isin([0, 3, 4])]
df.loc[df["y"] != 0, "y"] = 1

# Load train test dataframe
df_list_train = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_train.csv").iloc[:, 1]
df_list_test  = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\batch_3c\list_tumbon_test.csv").iloc[:, 1]

# Create tambon label features
df_tambon = df.groupby(["final_plant_year", "tambon_pcode"]).agg(dict_agg_features)
df_tambon = df_tambon[~df_tambon.iloc[:, 1:].isna().any(axis=1)]
df_tambon = df_tambon.reset_index()

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
df_tambon_train = 
#%%
df_tambon["tambon_pcode"].isin(df_list_train).sum()
#%%
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#%%
pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
#%%
pipeline.fit(x_train, y_train)
pipeline.score(x_train, y_train)
pipeline.score(x_test, y_test)
#%%
from icedumpy.plot_tools import plot_roc_curve
fig, ax = plt.subplots()
plot_roc_curve(pipeline, x_train, y_train, label='Train', color="g-", ax=ax)
plot_roc_curve(pipeline, x_test, y_test, label='Test', color="b-", ax=ax)
#%%
feature_importances = dict(zip(df_tambon_level.columns[1:], pipeline[1].feature_importances_))
#%%
import pandas as pd
df = pd.read_parquet(r"F:\CROP-PIER\CROP-WORK\batch_3c\df_pierxda_batch_3c_NE2.parquet")
#%%

















































