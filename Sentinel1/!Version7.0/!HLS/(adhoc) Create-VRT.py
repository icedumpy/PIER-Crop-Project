import os
import datetime
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.geo_tools import create_vrt
#%%
root_raster = r"F:\CROP-PIER\CROP-WORK\!hls-prep\ndvi-16d-noise-reduct-splitted"
root_save = r"C:\Users\PongporC\Desktop\temp\HLS-NDVI"
#%%
# list_info = []
# for pathrow in os.listdir(root_raster):
#     for year in os.listdir(os.path.join(root_raster, pathrow)):
#         for file in os.listdir(os.path.join(root_raster, pathrow, year)):
#             if not file.endswith(".tif"):
#                 continue
#             path_raster = os.path.join(root_raster, pathrow, year, file)
#             list_info.append({
#                 "pathrow":pathrow,
#                 "year":year,
#                 "date":datetime.datetime.strptime(file.split(".tif")[0][-10:], "%Y-%m-%d"),
#                 "filename":file,
#                 "path_raster":path_raster
#             })
# #%%
# df_info = pd.DataFrame(list_info)    
# for (pathrow, year), df_grp in df_info.groupby(["pathrow", "year"]):
#     df_grp = df_grp.sort_values(by="date")
#     os.makedirs(os.path.join(root_save, pathrow), exist_ok=True)
#     create_vrt(
#         path_save=os.path.join(root_save, pathrow, f"{pathrow}_{year}.vrt"),
#         list_path_raster=df_grp["path_raster"].tolist(), 
#         list_band_name=df_grp["date"].astype("str").tolist(),
#     )
#%% Sava image
for pathrow in os.listdir(root_raster):
    for year in os.listdir(os.path.join(root_raster, pathrow)):
        for file in os.listdir(os.path.join(root_raster, pathrow, year)):
            if not file.endswith(".tif"):
                continue
            os.makedirs(os.path.join(root_save, "PNG", pathrow), exist_ok=True)
            path_raster = os.path.join(root_raster, pathrow, year, file)
            raster = rasterio.open(path_raster).read(1)
            plt.figure(figsize=(9, 9))
            plt.imshow(raster, vmin=0, vmax=1, cmap="gray")
            plt.colorbar()
            plt.title(f"{pathrow}_{file[-14:-4]}")
            plt.savefig(os.path.join(root_save, "PNG", pathrow, f"{file.replace('.tif', '.png')}"), bbox_inches="tight")
            plt.close("all")