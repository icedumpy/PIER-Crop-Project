import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
root = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated\s1ab_vew_plant_info_official_polygon_disaster_all_rice_by_year_temporal(at-False)"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210809"
#%%
list_dict = []
for file in os.listdir(root):
    print(file)
    path_file = os.path.join(root, file)
    df = pd.read_parquet(path_file)
    dict_size = df.groupby("ext_act_id").size().value_counts().to_dict()
    list_dict.append(dict_size)
#%%
dict_n_pixels = dict()
for item in list_dict:
    for key in item.keys():
        if not key in dict_n_pixels.keys():
            dict_n_pixels[key] = item[key]
        else:
            dict_n_pixels[key] += item[key]
df = pd.DataFrame.from_dict([dict_n_pixels]).T
df = df.sort_index()
df.columns = ["n_pixels"]
df["n_pixels_normalized"] = df["n_pixels"]/df["n_pixels"].sum()
df = df.iloc[0:25]
#%%
plt.close("all")
plt.figure()
df["n_pixels_normalized"].plot.bar()
plt.xticks(rotation=0)
plt.ylabel("Probability")
plt.xlabel("n pixels")
plt.savefig(os.path.join(root_save, "n_pixels_histogram.png"), bbox_inches="tight")

plt.figure()
df["n_pixels_normalized"].cumsum().plot.bar()
plt.xticks(rotation=0)
plt.ylabel("Probability")
plt.xlabel("n pixels")
plt.savefig(os.path.join(root_save, "n_pixels_cumsum_histogram.png"), bbox_inches="tight")
