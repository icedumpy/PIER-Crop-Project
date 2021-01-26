import os
import datetime
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump, load
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def save_model(model, path):
    dump(model, path) 
    
def load_model(path):
    return load(path) 

def get_raster_date(raster, dtype):
    if dtype=='rasterio':
        raster_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
    elif dtype=='gdal':
        try:
            raster_date = [datetime.datetime(int(raster.GetRasterBand(i+1).GetDescription()[-8:-4]), int(raster.GetRasterBand(i+1).GetDescription()[-4:-2]), int(raster.GetRasterBand(i+1).GetDescription()[-2:])) for i in range(raster.RasterCount)]
        except:
            raster_date = [datetime.datetime(int(raster.GetRasterBand(i+1).GetDescription().split('-')[0]), int(raster.GetRasterBand(i+1).GetDescription().split('-')[1]), int(raster.GetRasterBand(i+1).GetDescription().split('-')[2])) for i in range(raster.RasterCount)]
        
    return np.array(raster_date)

def modeFilter(im, mode_size, ratio):
    '''
    Mode filter function
    where
        'im' : input image
        'mode_size' : filter size
    '''
    kernel = np.ones((mode_size, mode_size), 'float32')
    out = cv2.filter2D(im, ddepth=cv2.CV_8UC1, kernel=kernel)
    th = mode_size**2-1
#    th = th//2
    th = int(th*ratio)
    out = (out>th).astype('uint8')
    return out

def create_tiff(path_save, im, projection, geotransform, list_band_name, nodata = -9999, dtype=gdal.GDT_Byte, channel_first=True):
    if len(im.shape)==2:
        im = np.expand_dims(im, 0)
    if not channel_first:
        im = np.moveaxis(im, 0, -1)
    
    band = im.shape[0]
    row = im.shape[1]
    col = im.shape[2]
    driver = gdal.GetDriverByName('ENVI')
    output = driver.Create(path_save, col, row, band, dtype)
    output.SetProjection(projection)
    output.SetGeoTransform(geotransform)
    for i in range(band):
        band_name = list_band_name[i]
        output.GetRasterBand(i+1).SetDescription(band_name)
        output.GetRasterBand(i+1).WriteArray(im[i, :, :])   
        output.GetRasterBand(i+1).SetNoDataValue(nodata)
        output.FlushCache()

    del output
    del driver
    
def sampling_dataset(root, flood = True, sampling_size = 0.1):
    list_df = []
    for filename in os.listdir(root):
        df = pd.read_parquet(os.path.join(root, filename))    
        
        # Clean: นาปีี
        df = df[((df['final_plant_date'].dt.month>=5) & (df['final_plant_date'].dt.month<=10))]
      
        # Clean: Drop nan
        df = df[~np.any(df[['t-2', 't-1', 't', 't+1']].isna(), axis=1)]
        
        if flood:
            # Clean: Drop low loss raio (only [0.8, 1] remain)
            df = df[(df['loss_ratio']>=0.8) & (df['loss_ratio']<=1)]
        
        # Clean 3: Drop duplicate
        df = df.drop_duplicates(subset=['t-2', 't-1', 't', 't+1'])
    
        # Sampling 10%
        df = df.sample(frac=sampling_size)
        list_df.append(df)

    df = pd.concat(list_df, ignore_index=True)
    return df

def load_dataset_df(root_df_dry, root_df_flood, index_name, sampling_size = 1):
    '''
    Load dataset for model
            root_df_dry: directory of df_dry
            root_df_flood: directory of df_flood
            index_name: selecting condition ex. 's104', p'45', 'all'
            sampling_size: if index_name = 'all', samping size of each file
    
    '''
    if index_name == 'all':
        df_flood = sampling_dataset(root_df_flood, flood = True, sampling_size = 1)
        df_dry = sampling_dataset(root_df_dry, flood = False, sampling_size = 1)
    else:
        df_flood = pd.concat([pd.read_parquet(os.path.join(root_df_flood, file)) for file in os.listdir(root_df_flood) if index_name in file], ignore_index=True)
        df_dry = pd.concat([pd.read_parquet(os.path.join(root_df_dry, file)) for file in os.listdir(root_df_dry) if index_name in file], ignore_index=True)
        
        # Clean: Drop นาปรัง
        df_flood = df_flood[((df_flood['final_plant_date'].dt.month>=5) & (df_flood['final_plant_date'].dt.month<=10))]
        df_dry = df_dry[((df_dry['final_plant_date'].dt.month>=5) & (df_dry['final_plant_date'].dt.month<=10))]
        
        # Clean: Drop nan
        df_flood = df_flood[~np.any(df_flood[['t-2', 't-1', 't', 't+1']].isna(), axis=1)]
        df_dry = df_dry[~np.any(df_dry[['t-2', 't-1', 't', 't+1']].isna(), axis=1)]
        
        # Clean: Drop low loss raio (only [0.8, 1] remain)
        df_flood = df_flood[(df_flood['loss_ratio']>=0.8) & (df_flood['loss_ratio']<=1)]
        
        # Clean 3: Drop duplicate
        df_flood = df_flood.drop_duplicates(subset=['t-2', 't-1', 't', 't+1'])
        df_dry = df_dry.drop_duplicates(subset=['t-2', 't-1', 't', 't+1'])
    
    return df_dry, df_flood

def plot_roc_curve(model, x, y, label, color='b-'):
    y_predict_prob = model.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predict_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color, label = f"{label} (AUC = {auc:.4f})")
    plt.plot(fpr, fpr, "--")
    plt.xlabel("False Alarm")
    plt.ylabel("Detection")
    
    return y_predict_prob, fpr, tpr, thresholds

def load_vew_mapping_df(root_mapping, root_vew, index_name, date_range):
    # Load mapping
    df_mapping = pd.concat([pd.read_parquet(os.path.join(root_mapping, file)) for file in os.listdir(root_mapping) if index_name in file]) 
    list_p_code = [file[28:31] for file in os.listdir(root_mapping) if index_name in file]
    
    # Load vew
    df_vew = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in os.listdir(root_vew) if os.path.splitext(file)[0][-3:] in list_p_code], ignore_index=True)
    df_vew = df_vew[df_vew['new_polygon_id'].isin(df_mapping.index)]
    
    # Remove zero plant area
    df_vew = df_vew[df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']!=0]
    
    # Get Loss ratio
    df_vew['loss_ratio'] = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA'])

    # Drop invalid final_plant_date
    df_vew = df_vew.dropna(subset=['final_plant_date'])
    
    # Select only Normal and Flood
    df_vew = df_vew[(df_vew['DANGER_TYPE_NAME']=='อุทกภัย') | (pd.isnull(df_vew['DANGER_TYPE_NAME']))]
    
    # Convert string into datetime
    df_vew['final_plant_date'] = pd.to_datetime(df_vew['final_plant_date'])
    
    # Replace no START_DATAE data with NaT
    df_vew.loc[df_vew[pd.isnull(df_vew['START_DATE'])].index, 'START_DATE'] = pd.NaT
    df_vew['START_DATE'] = pd.to_datetime(df_vew['START_DATE'],  errors='coerce')

    # Drop out of range date : final_plant_date in range or start_date in range
    df_vew = df_vew[((df_vew['final_plant_date']>=date_range[0]) & (df_vew['final_plant_date']<=date_range[1])) | ((df_vew['START_DATE']>=date_range[0]) & (df_vew['START_DATE']<=date_range[1]))]
    index_to_drop = df_vew[~pd.isnull(df_vew['START_DATE'])][df_vew[~pd.isnull(df_vew['START_DATE'])]['START_DATE'] > date_range[-1]].index.to_list()
    if len(index_to_drop)!=0:
        df_vew.drop(index = index_to_drop, inplace=True)

    # Drop case [flood and loss_ratio==0] :== select only [not flood or loss_ratio!=0]
    df_vew = df_vew[(df_vew['loss_ratio']!=0) | pd.isna(df_vew['START_DATE'])]
    
    # Drop out of range loss ratio
    df_vew = df_vew[(df_vew['loss_ratio']>=0) & (df_vew['loss_ratio']<=1)]
    
    # นาปี
    df_vew = df_vew[((df_vew['final_plant_date'].dt.month>=5) & (df_vew['final_plant_date'].dt.month<=10))]
    df_vew = df_vew.reset_index(drop=True)    

    return df_vew, df_mapping

def create_geodataframe(df, polygon_column='final_polygon', crs={'init':'epsg:4326'}):
    gdf = gpd.GeoDataFrame(df)
    gdf['geometry'] = gdf[polygon_column].apply(wkt.loads)
    gdf.crs = crs
    return gdf


def create_rice_mask(path_shp, root_mapping, root_vew, index_name, raster, date_range):
    # Create folder
    if not os.path.exists(os.path.dirname(path_shp)):
        os.makedirs(os.path.dirname(path_shp))
    
    # Load vew of selected index ('s403')
    df_vew, _ = load_vew_mapping_df(root_mapping, root_vew, index_name, date_range)
    
    # Convert dataframe to geodataframe
    gdf_vew = create_geodataframe(df_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'loss_ratio', 'final_polygon']])
    gdf_vew['START_DATE'] = gdf_vew['START_DATE'].astype(str)
    gdf_vew['final_plant_date'] = gdf_vew['final_plant_date'].astype(str)
    gdf_vew = gdf_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'loss_ratio', 'geometry']]
    
    # Save geodataframe to shapefile
    gdf_vew.to_file(path_shp)
    
    # Rasterize shapefile to raster 
    path_rasterize = os.path.join(os.path.dirname(path_shp), os.path.basename(path_shp).replace('.shp', '.tif'))
    command = f"gdal_rasterize -at -l {os.path.splitext(os.path.basename(path_rasterize))[0]} -burn 1.0 -ts {raster.RasterXSize:.1f} {raster.RasterYSize:.1f} -a_nodata 0.0 -te {raster.GetGeoTransform()[0]} {raster.GetGeoTransform()[3] + raster.RasterYSize*raster.GetGeoTransform()[5]} {raster.GetGeoTransform()[0] + raster.RasterXSize*raster.GetGeoTransform()[1]} {raster.GetGeoTransform()[3]} -ot Byte -of GTiff" + " {} {}".format(path_shp.replace('\\', '/'), path_rasterize.replace('\\', '/'))
    os.system(command)
    return 1
    
def get_fuzzy_confusion_matrix(predicted, ref):
    '''
    This function return fuzzy confusion matrix
    where
        predicted is list of predicted loss ratio, ex. [0.5, 0.0, 1.0, 0.8, 0.6]
        ref is list of groundtruth loss ratio,     ex. [0.7, 1.0, 0.5, 0.4, 0.6]
    '''
    '''
                        Ref
                   |Flooded|No Flood|
    Pred    Flooded|_______|________|
         No Flooded|_______|________|
    '''
    f_cnf_matrix = np.zeros((2, 2), dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    df_results = pd.DataFrame({'Pred_flooded' : predicted, 'Ref_flooded' : ref, 'Pred_no_flood' : 1-predicted, 'Ref_no_flood' : 1-ref})

    # Add data in (Pred_flooded, Ref_flooded)
    f_cnf_matrix[0, 0] = df_results[['Pred_flooded', 'Ref_flooded']].min(axis=1).sum()
    
    # Add data in (Pred_flooded, Ref_no_flood)
    f_cnf_matrix[0, 1] = df_results[['Pred_flooded', 'Ref_no_flood']].min(axis=1).sum()
    
    # Add data in (Pred_no_flood, Ref_flooded)
    f_cnf_matrix[1, 0] = df_results[['Pred_no_flood', 'Ref_flooded']].min(axis=1).sum()
    
    # Add data in (Pred_no_flood, Ref_no_flood)
    f_cnf_matrix[1, 1] = df_results[['Pred_no_flood', 'Ref_no_flood']].min(axis=1).sum()

#    OA = 100*np.diag(f_cnf_matrix).sum()/len(predicted)
#    DT = 100*f_cnf_matrix[0, 0]/(f_cnf_matrix[0, 0]+f_cnf_matrix[1, 0]) # Recall, TPR
#    FA = 100*f_cnf_matrix[0, 1]/(f_cnf_matrix[0, 1]+f_cnf_matrix[1, 1]) # FPR

    return f_cnf_matrix
    
def train_model(df_dry, df_flood, balanced = True, sample_size = None, verbose=1):
    '''
        Train model
            df_dry: dry dataframe
            df_flood: flood dataframe
            balanced: balance or not
            sample_size: if balanced==True, sample size if the size of each class (>=1)
                         if balanced==False, sample size if the fraction of each class (<=1) 
        
    '''
    tqdm.write("Loading training dataset")
    if balanced:
        if sample_size == None:
            sample_size = min(len(df_dry), len(df_flood))
        samples_dry = df_dry[['t-2', 't-1', 't', 't+1']].sample(n=sample_size)
        samples_flood = df_flood[['t-2', 't-1', 't', 't+1']].sample(n=sample_size)
        
    else:
        if sample_size == None:
            sample_size = 1
        samples_dry = df_dry[['t-2', 't-1', 't', 't+1']].sample(frac=sample_size)
        samples_flood = df_flood[['t-2', 't-1', 't', 't+1']].sample(frac=sample_size)
    
    x = np.concatenate((samples_flood.values, samples_dry.values), axis=0)
    y = np.zeros(len(x))
    y[:len(samples_flood)] = 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    
    tqdm.write("Start training")
    model = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=4, random_state=0, verbose=verbose, n_jobs=-1)
    model.fit(x_train, y_train)
    
    return model, [x_train, x_test, y_train, y_test]

def create_flood_map(path_save, model, path_raster, path_mask, root_mapping, root_vew, index_name):
    raster = gdal.Open(path_raster)
    raster_date = get_raster_date(raster, dtype='gdal')    
    
    tqdm.write(f"Loading rice mask")
    if not os.path.exists(path_mask):
        tqdm.write(f"No available rice mask")
        tqdm.write(f"Creating rice mask")
        create_rice_mask(path_mask, root_mapping, root_vew, index_name, raster, date_range = (raster_date[0], raster_date[-1]))
    mask_rice = gdal.Open(path_mask.replace('.shp', '.tif')).ReadAsArray()
    row_rice, col_rice = np.nonzero(mask_rice)    
    
    raster_im = raster.ReadAsArray()
    flood_im = np.zeros_like(raster_im, dtype='float32')
    flood_im = flood_im - 9999
    
    tqdm.write(f"Creating flood map")
    for band in tqdm(range(2, raster.RasterCount-2)):
        # Check if consecutive 4 weeks? muse be equal to 36
        if (raster_date[band+1]-raster_date[band-2]).days == 36:
            pred = model.predict_log_proba(raster_im[band-2:band+2, row_rice, col_rice].T)
            log_likelihood = pred[:,1]- pred[:,0]
            flood_im[band, row_rice, col_rice] = log_likelihood

    tqdm.write(f"Saving flood map")
    create_tiff(path_save = path_save, 
                im = flood_im, 
                projection = raster.GetProjection(), 
                geotransform = raster.GetGeoTransform(), 
                list_band_name = [str(item.date()) for item in raster_date], 
                nodata= -9999,
                dtype = gdal.GDT_Float32, 
                channel_first=True)
    
    return flood_im, raster_date






