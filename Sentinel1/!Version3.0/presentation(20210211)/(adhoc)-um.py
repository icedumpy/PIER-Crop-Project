import pandas as pd
df_sat = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\df_satellite_vs_doae_end_season.csv")
df_drone = pd.read_csv(r"F:\CROP-PIER\CROP-WORK\pct_damge_drone_doae.csv")
df_drone = df_drone[df_drone.columns[1:]]
df_drone.columns = ["activity_id"] + df_drone.columns[1:].tolist()
df_drone[df_drone.columns[1:]] = 100*df_drone[df_drone.columns[1:]].astype("float32")
#%%
df = pd.merge(df_sat, df_drone, on=["activity_id"], how="inner")
#%%
df["DOAE"] = df["pct_damage_doae"] > 0
df["Sat"] = df["combined_predict_f"] > 0
df["Drone"] = df["pct_damage_drone"] > 0
#%%
df["1"] = df["DOAE"] & df["Sat"] & df["Drone"]
df["2"] = df["Sat"] & df["Drone"]
df["3"] = df["Sat"] & df["DOAE"]
df["4"] = df["Drone"] & df["DOAE"]
df["5"] = df["Sat"] & (~df["DOAE"]) & (~df["Drone"])
df["6"] = df["Drone"] & (~df["DOAE"]) & (~df["Sat"])
df["7"] = df["DOAE"] & (~df["Sat"]) & (~df["Drone"])







