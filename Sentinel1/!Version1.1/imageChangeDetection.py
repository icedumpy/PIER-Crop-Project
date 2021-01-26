import os
import sys
import json
from osgeo import gdal, osr
import numpy as np
from skimage import filters
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(r"F:\CROP-PIER\CROP-WORK\code")
from general_fn import *

def modeFilter(im, size):
    kernel = np.ones((size, size), 'float32')
    out = cv2.filter2D(im, ddepth=cv2.CV_8UC1, kernel=kernel)
    th = size**2-1
    th = th//2
    out = (out>th).astype('uint8')
    return out

def creat_flood_raster(path_raster, root_save, threshold_data, video=False, selected_band = (0, 0)):
    raster = gdal.Open(path_raster)

    # Initialize parameters
    strip_id = threshold_data['raster_strip_id']
    water_thresh = threshold_data['water_threshold']
    to_water_thresh =  threshold_data['change_to_water_threshold']
    to_land_thresh =  threshold_data['change_from_water_threshold']  
    
    if selected_band!=(0, 0):
        num_bands = selected_band[1] - selected_band[0] + 1 
        selected_band = (selected_band[0]-1, selected_band[1])
    else:
        num_bands = raster.RasterCount
    width = raster.RasterXSize
    height = raster.RasterYSize    
    valid_mask = np.ones((height, width), 'uint8')
    
    # Create empty video file
    if video:
        frame_width = width//5
        frame_height = height//5
        out_video = cv2.VideoWriter(os.path.join(root_save,  f"Flood_VV_S{strip_id}.avi"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (frame_width*2, frame_height))
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get valid_mask, and list of band_date
    date_list = []
    for band in tqdm(range(num_bands)):
        raster_band = raster.GetRasterBand(selected_band[0]+band+1)
        band_name = raster_band.GetDescription()
        date = band_name[-8:]
        date_list.append(date)
        
        raster_im = raster_band.ReadAsArray()
        valid_mask *= (raster_im>0)
    
    row_valid, col_valid = np.nonzero(valid_mask)
    row_nvalid, col_nvalid = np.nonzero(valid_mask==0)
    
    # Create empty raster
    driver = gdal.GetDriverByName('GTiff')
    out_image = driver.Create(os.path.join(root_save, f"Flood_VV_S{strip_id}.tiff"), xsize=width, ysize=height, bands=num_bands, eType=gdal.GDT_Byte)
    src = osr.SpatialReference()
    src.ImportFromWkt(raster.GetProjection())
    out_image.SetProjection(src.ExportToWkt())
    out_image.SetGeoTransform(raster.GetGeoTransform())
    out_image.SetMetadata(raster.GetMetadata())
    
    # Create flood map
    flood_map = np.zeros((height, width), np.uint8)
    im_previous = None
    for band in tqdm(range(num_bands)):
        raster_band = raster.GetRasterBand(selected_band[0]+band+1)
        tqdm.write(f"START: {raster_band.GetDescription()}")
        out_band = out_image.GetRasterBand(band+1)
        
        raster_im = raster_band.ReadAsArray()
        out_im = (raster_im<water_thresh).astype('uint8')
        
        if im_previous is not None:
            diff_im = im_previous - raster_im    
            changed_to_water_area = (diff_im>to_water_thresh).astype('uint8')
            
            # Pixel that change from land to water == 1 
            flood_map += changed_to_water_area
            changed_from_water_area = (diff_im<to_land_thresh).astype('uint8')
            
            # Find and remove flood's pixels that change from water to land and already land (from otsu)
            row_not_flood, col_not_flood = np.nonzero((changed_from_water_area)+(out_im==0)) # (change from water to land map + not water map))
            flood_map[row_not_flood, col_not_flood] = 0
            flood_map[row_nvalid, col_nvalid] = 0     
        
        # Save flood map and continue for next loop
        im_previous = raster_im.copy()
        out_im = (flood_map>0).astype('uint8')
#        out_im = modeFilter(flood_map, size=5)
        out_band.WriteArray(out_im)
        out_band.SetDescription(date_list[band])
        out_band.FlushCache()
        
        # Write 1 frame
        if video:
            out_im *= 255   
            
            out_im[row_nvalid, col_nvalid] = 0
            
            out_im2 = cv2.resize(out_im, (frame_width, frame_height))
            out_im3 = np.zeros((frame_height, frame_width, 3), 'uint8')
            
            raster_im_resize = cv2.resize(raster_im, (frame_width, frame_height))
            vmin = np.percentile(raster_im_resize, 2)
            vmax = np.percentile(raster_im_resize, 98)
            
            raster_im_resize -= vmin
            raster_im_resize /= vmax
            
            raster_im_resize = 255.0*(raster_im_resize>1.0) + 255.0*raster_im_resize*(raster_im_resize<=1)*(raster_im_resize>=0)
            raster_im_resize_jet = cv2.applyColorMap(raster_im_resize.astype('uint8'), cv2.COLORMAP_JET)
            
            out_im3[:, :, 0] = out_im2
            out_im3[:, :, 1] = out_im2
            out_im3[:, :, 2] = out_im2
            cv2.putText(out_im3, date_list[band] + "_{}".format(band+1), (10, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("out_th", out_im3)
            cv2.imshow("raster_im_resize_jet", raster_im_resize_jet)
            cv2.waitKey(10)
            out_im3 = np.concatenate((out_im3, raster_im_resize_jet), axis=1)
            out_video.write(out_im3)
    if video:
        out_video.release()        
    out_image = None

def load_json(path):
    return json.load(open(path, 'r'))
#%% Initial parameters
strip_id = 304
#root_save = r"F:\CROP-PIER\CROP-WORK\Flood-Map"
root_save = r"C:\Users\PongporC\Desktop\adhoc"
#root_save = r"F:\CROP-PIER\CROP-WORK\Sentinel1-new-image"
path_raster = r"F:\CROP-PIER\CROP-WORK\Complete_VV\LSVVS{0:d}_2017-2019".format(strip_id)
#path_raster = r"F:\CROP-PIER\CROP-WORK\Sentinel1-new-image\new.vrt"

path_threshold = r"F:\CROP-PIER\CROP-WORK\Flood-evaluation\\{0:d}\Flood_VV_S{1:d}_(0.70,0.30,0.70).json".format(strip_id, strip_id)
#%%
threshold_data = load_json(path_threshold)
#%%
creat_flood_raster(path_raster, root_save, threshold_data, video=True, selected_band = (0, 0))
#%%
