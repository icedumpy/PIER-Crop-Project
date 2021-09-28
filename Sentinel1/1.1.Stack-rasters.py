#%% Import libraries
import os
import datetime
import pandas as pd
from osgeo import gdal
#%% Define functions
raster_datetime = lambda file: datetime.datetime.strptime(os.path.splitext(file)[0].split("_")[1], "%Y-%m-%d")
path_row = lambda file, root_raster_mapping: os.path.join(root_raster_mapping, f"s1_pixel_row_map_{os.path.basename(os.path.dirname(file))}.tiff")
path_col = lambda file, root_raster_mapping: os.path.join(root_raster_mapping, f"s1_pixel_col_map_{os.path.basename(os.path.dirname(file))}.tiff")

def get_raster_files_info(root_raster, list_ignored):
    list_path = []
    list_date = []
    list_strip_id = []
    list_sat_type = []
    for root, _, files in os.walk(root_raster):
        for idx, file in enumerate(files):
            if file.endswith(tuple(list_ignored)):
                continue
            path_file = os.path.join(root, file)
            strip_id = os.path.basename(root)
            sat_type = os.path.basename(os.path.dirname(root))
            
            date = raster_datetime(file)
            list_path.append(path_file)
            list_date.append(date)
            list_strip_id.append(strip_id)
            list_sat_type.append(sat_type)
            
    df = pd.DataFrame(
            {"path_file":list_path,
             "date":list_date,
             "strip_id":list_strip_id,
             "sat_type":list_sat_type}, columns=["path_file", "date", "strip_id", "sat_type"]
    )
    
    df = df.sort_values(["date"])
    return df

def filter_raster_info(df, strip_id, sat_type):
    df = df.copy()
    df = df[df["strip_id"] == strip_id]
    if sat_type != "S1AB":
        df = df[df["sat_type"] == sat_type]
    return df

def create_vrt(path_vrt, list_path_raster, list_name=None):
    list_path_raster = [path_row(list_path_raster[0], root_raster_mapping), 
                        path_col(list_path_raster[0], root_raster_mapping)]  + list_path_raster
    outds = gdal.BuildVRT(path_vrt, list_path_raster, separate=True)
    if list_name is not None:
        list_name = ["row", "col"] + list_name
        for channel, name in zip(range(1, outds.RasterCount+1), list_name):
            outds.GetRasterBand(channel).SetDescription(name)
    outds.FlushCache()
    del outds

def create_trs(path_dst, path_src, file_format='ENVI'):
    outds = gdal.Translate(path_dst, path_src, format=file_format)
    outds.FlushCache()
    del outds

def main(strip_id, sat_type, trs=False):
    path_vrt = os.path.join(root_temp, sat_type, f"{sat_type}_{strip_id}.vrt")
    path_trs = os.path.join(root_temp, sat_type, f"{sat_type}_{strip_id}")
    os.makedirs(os.path.dirname(path_vrt), exist_ok=True)
    
    df_info = get_raster_files_info(root_raster, list_ignored)
    df_info_filtered = filter_raster_info(df_info, strip_id=strip_id, sat_type=sat_type)
    if sat_type == "S1AB":
        date_s1a = df_info_filtered.loc[df_info_filtered["sat_type"] == "S1B", "date"].min() - datetime.timedelta(days=6)
        df_info_filtered = df_info_filtered[df_info_filtered["date"] >= date_s1a]
    
    create_vrt(path_vrt, df_info_filtered["path_file"].tolist(), (df_info_filtered["date"].dt.date.astype(str).str.split("-").str.join("") + "_" + df_info_filtered["sat_type"]).tolist())
    if trs:
        create_trs(path_trs, path_vrt, file_format='ENVI')
#%%
if __name__ == "__main__":
    root_raster_mapping = r"G:\!PIER\!FROM_2TB\s1_pixel_rowcol_map"
    root_temp = r"C:\Users\PongporC\Desktop\temp"
    root_raster = r"G:\!PIER\!FROM_2TB\Complete_VV_separate"
    list_ignored = [".xml", ".ini"]
    
    for strip_id in ["101", "102", "103", "104", "105", "106", "107", "108", "109", 
                     "201", "202", "203", "204", "205", "206", "207", "208",
                     "301"]:
        print("S1A", strip_id)
        sat_type = "S1A"
        main(strip_id, sat_type)
        
    for strip_id in ["302", "303", "304", "305", "306", "401", "402", "403"]:
        print("S1AB", strip_id)
        main(strip_id, sat_type="S1B")
        main(strip_id, sat_type="S1AB")
#%%