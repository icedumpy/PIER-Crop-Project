import os
import numpy as np
import pandas as pd
from icedumpy.geo_tools import convert_to_geodataframe
from multiprocessing import Pool

# root = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
# root_save = r"F:\CROP-PIER\CROP-WORK\vew_shp"
root = r"F:\CROP-PIER\CROP-WORK\vew_2020\vew_polygon_id_plant_date_disaster_20210202"
root_save = r"F:\CROP-PIER\CROP-WORK\vew_shp_2020"
new_columns = ['ext_act_id', 'AREA_WA',
               'plant_date', 'START_DATE', 'DANG_NAME',
               'D_AREA_WA', "loss_ratio", 'geometry']

def worker(file):
    df = pd.read_parquet(os.path.join(root, file))
    df.loc[df['DANGER_TYPE_NAME'] == "ฝนทิ้งช่วง", "DANGER_TYPE_NAME"] = "ภัยแล้ง" 
    
    # Select only no loss, flood, drought
    df = df[(pd.isna(df['START_DATE'])) |
            (df['DANGER_TYPE_NAME'] == "อุทกภัย") | 
            (df['DANGER_TYPE_NAME'] == "ภัยแล้ง")]
    
    # Calculate loss_ratio
    df = df.assign(loss_ratio = np.where(pd.isna(df['TOTAL_DANGER_AREA_IN_WA']/df['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df['TOTAL_DANGER_AREA_IN_WA']/df['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
    
    
    df["final_plant_date"] = df['final_plant_date'].astype(str)
    df["final_harvest_date"] = df['final_harvest_date'].astype(str)
    df["START_DATE"] = df['START_DATE'].astype(str)
    
    gdf = convert_to_geodataframe(df)
    gdf = gdf[gdf['geometry'].type == "MultiPolygon"]
    gdf = gdf[['ext_act_id', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA',
               'final_plant_date', 'START_DATE', 'DANGER_TYPE_NAME',
               'TOTAL_DANGER_AREA_IN_WA', "loss_ratio", 'geometry']]
    
    # Save all shapefiles based on final_plant_date (only no loss)
    for final_plant_date, gdf_grp in gdf[gdf['START_DATE'] == "NaT"].groupby("final_plant_date"):
        path_save = os.path.join(root_save, f'p{file.split(".")[0][-2:]}', "Normal", f"{final_plant_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        gdf_grp.columns = new_columns
        gdf_grp.to_file(path_save)
    
    # Save flood shapefiles based on START_DATE (only flood)
    for start_date, gdf_grp in gdf[gdf['DANGER_TYPE_NAME'] == "อุทกภัย"].groupby("START_DATE"):
        path_save = os.path.join(root_save, f'p{file.split(".")[0][-2:]}', "Flood", f"{start_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        gdf_grp.columns = new_columns
        gdf_grp.to_file(path_save)
        
    # Save drought shapefiles based on START_DATE (only drought)
    for start_date, gdf_grp in gdf[gdf['DANGER_TYPE_NAME'] == "ภัยแล้ง"].groupby("START_DATE"):
        path_save = os.path.join(root_save, f'p{file.split(".")[0][-2:]}', "Drought", f"{start_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        gdf_grp.columns = new_columns
        gdf_grp.to_file(path_save)
#%%
if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2) # Create a multiprocessing Pool
    pool.map(worker, os.listdir(root))  # process data_inputs iterable with pool
