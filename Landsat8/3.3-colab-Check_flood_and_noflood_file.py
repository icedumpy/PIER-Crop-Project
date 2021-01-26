import os
import numpy as np
import pandas as pd
#%%
root = r"F:\CROP-PIER\CROP-WORK\Landsat8_dataframe\ls8_noflood_value"
#%%
df = pd.DataFrame(columns=["path_file", "p", "pathrow", "band", "size(MB)"])
for file in os.listdir(root):
    df = df.append(pd.Series({"path_file" : os.path.join(root, file),
                              "p" : file.split(".")[0].split("_")[-3],
                              "pathrow" : file.split(".")[0].split("_")[-2],
                              "band" : file.split(".")[0].split("_")[-1],
                              "size(MB)" : os.path.getsize(os.path.join(root, file))/1024/1024}),
                   ignore_index=True)
df = df[df['band']!='BQA']
#%%
total = 0
list_to_remove = []
for (p, pathrow), df_grp in df.groupby(['p', 'pathrow']):
   df_grp['size(MB)'].std()
   mse = (np.power(df_grp['size(MB)'].mean()-df_grp['size(MB)'], 2)).sum()/len(df_grp)
   if mse>1:
       total+=1
       print(df_grp)
       list_to_remove += df_grp['path_file'].tolist()
print("Total:", total)
#%%
for path in list_to_remove:
    os.remove(path)
#%%
df = pd.DataFrame(columns=["path_file", "p", "pathrow", "band", "size(MB)"])
for file in os.listdir(root):
    df = df.append(pd.Series({"path_file" : os.path.join(root, file),
                              "p" : file.split(".")[0].split("_")[-3],
                              "pathrow" : file.split(".")[0].split("_")[-2],
                              "band" : file.split(".")[0].split("_")[-1],
                              "size(MB)" : os.path.getsize(os.path.join(root, file))/1024/1024}),
                   ignore_index=True)
#%%
list_to_remove = []
for (p, pathrow), df_grp in df.groupby(['p', 'pathrow']):
    if len(df_grp)<9:
        list_to_remove += df_grp['path_file'].tolist()
#%%
for path in list_to_remove:
    os.remove(path)