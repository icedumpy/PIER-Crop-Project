import os
import pandas as pd
import icedumpy
from multiprocessing import Pool

root = r"F:\CROP-PIER\CROP-WORK\vew_polygon_id_plant_date_disaster_merged"
root_save = r"F:\CROP-PIER\CROP-WORK\Model visualization\vew_shp"
new_columns = ['new_pol_id', 'AREA_WA',
               'plant_date', 'START_DATE', 'DANG_NAME',
               'D_AREA_WA', 'geometry']

def worker(file):
    df = pd.read_parquet(os.path.join(root, file))
    
    # Select only no flood and flood
    df = df[(pd.isna(df['START_DATE'])) | (df['DANGER_TYPE_NAME'] == "อุทกภัย")]
    
    # Select only plant date between 2017-2019
    df = df.loc[df['final_plant_date'].apply(lambda val: int(val[:4]) in range(2017, 2020))]
    df["START_DATE"] = df['START_DATE'].astype(str)
    
    gdf = icedumpy.geo_tools.convert_to_geodataframe(df)
    gdf = gdf[gdf['geometry'].type == "MultiPolygon"]
    gdf = gdf[['new_polygon_id', 'TOTAL_ACTUAL_PLANT_AREA_IN_WA',
               'final_plant_date', 'START_DATE', 'DANGER_TYPE_NAME',
               'TOTAL_DANGER_AREA_IN_WA', 'geometry']]

    # Save all shapefiles based on final_plant_date (flood and no flood)
    for final_plant_date, gdf_grp in gdf.groupby("final_plant_date"):
        path_save = os.path.join(root_save, file.split(".")[0][-3:], "final_plant_date", f"{final_plant_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        
        gdf_grp.columns = new_columns
        gdf_grp.to_file(path_save)
    
    # Save flood shapefiles based on START_DATE (only flood)
    for start_date, gdf_grp in gdf[gdf['DANGER_TYPE_NAME'] == "อุทกภัย"].groupby("START_DATE"):
        path_save = os.path.join(root_save, file.split(".")[0][-3:], "start_date", f"{start_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        
#        tqdm.write(f"Saving (start_date): {start_date}({len(gdf_grp)})_{file.split('.')[0].split('_')[-1]}.shp")
        gdf_grp.columns = new_columns
        gdf_grp.to_file(path_save)
#%%
if __name__ == '__main__':
    pool = Pool(os.cpu_count()-1) # Create a multiprocessing Pool
    pool.map(worker, os.listdir(root))  # process data_inputs iterable with pool
