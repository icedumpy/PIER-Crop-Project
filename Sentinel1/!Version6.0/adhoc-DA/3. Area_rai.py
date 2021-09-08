import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
root = r"F:\CROP-PIER\CROP-WORK\vew_plant_info_official_polygon_disaster_all_rice_by_year"
root_save = r"F:\CROP-PIER\CROP-WORK\Presentation\20210809"
#%%
list_rai = []
for file in os.listdir(root):
    print(file)
    path_file = os.path.join(root, file)
    df = pd.read_parquet(path_file)
    list_rai.append((0.0025*df["TOTAL_ACTUAL_PLANT_AREA_IN_WA"]).tolist())
list_rai = [item2 for item1 in list_rai for item2 in item1]
#%%
df = pd.DataFrame(list_rai)
df = df[df[0] <= df[0].quantile(0.99)]
df = df.sample(frac=0.1)
df.columns = ["Area(Rai)"]
#%%
sns.histplot(data=df, x="Area(Rai)", bins=30, stat="probability")
plt.savefig(os.path.join(root_save, "area_rai.png"), bbox_inches="tight")
