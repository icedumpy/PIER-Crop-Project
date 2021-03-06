from osgeo import gdal, osr
import numpy as np
from skimage import filters
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
def modeFilter(im, size):
    kernel = np.ones((size, size), 'float32')
    out = cv2.filter2D(im, ddepth=cv2.CV_8UC1, kernel=kernel)
    th = size**2-1
    th = th//2
    out = (out>th).astype('uint8')
    return out
#%% Initial parameters
video = True
#%%
im = gdal.Open(r"D:\Crop\Data\GIS\Sentinel-1\Complete_VV\LSVVS402_2017-2019")
num_bands = im.RasterCount
width = im.RasterXSize
height = im.RasterYSize
print(width, height)        
if video:
    frame_width = width//5
    frame_height = height//5
    out = cv2.VideoWriter("C:\Crop\data\Processed Data\Sentinel1_flood\Water_402.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (frame_width*2, frame_height))
    font = cv2.FONT_HERSHEY_SIMPLEX
#%%
valid = np.ones((height, width), 'uint8')
dates = []
for k in tqdm(range(num_bands)):
    b = im.GetRasterBand(k+1)
    dc = b.GetDescription()
    date = dc[-8:]
    dates.append(date)
    d = b.ReadAsArray()
    valid *= (d>0)
#%%
rw_v, cl_v = np.nonzero(valid)
rw_z, cl_z = np.nonzero(valid==0)
#%%
water = np.zeros((height, width), np.uint8)
driver = gdal.GetDriverByName('GTiff')
out_image = driver.Create(r"C:\Crop\data\Processed Data\Sentinel1_flood\WaterBody_402.tif", xsize=width, ysize=height, bands=num_bands, eType=gdal.GDT_Byte)
src = osr.SpatialReference()
src.ImportFromWkt(im.GetProjection())
out_image.SetProjection(src.ExportToWkt())
out_image.SetGeoTransform(im.GetGeoTransform())
out_image.SetMetadata(im.GetMetadata())
#%%
for k in tqdm(range(num_bands)):
#    tqdm.write("Working on Band {}".format(k+1))
    b = im.GetRasterBand(k+1)
    b_out = out_image.GetRasterBand(k+1)
    
    d = b.ReadAsArray()
    dd = d[rw_v, cl_v]
    
    idx = np.nonzero(dd < 0.05)[0]
    val = filters.threshold_otsu(dd[idx], nbins=4096)
    
    val = min(val, 0.028)
    out_im = (d<val).astype('uint8')
    out_im = modeFilter(out_im, size=7)
    b_out.WriteArray(out_im)
    b_out.SetDescription(dates[k])
    b_out.FlushCache()
    
    if video:
        out_im *= 255   
        
        out_im[rw_z, cl_z] = 0
        
        out_im2 = cv2.resize(out_im, (frame_width, frame_height))
        out_im3 = np.zeros((frame_height, frame_width, 3), 'uint8')
        
        dd = cv2.resize(d, (frame_width, frame_height))
        t1 = np.percentile(dd, 2)
        t2 = np.percentile(dd, 98)
        
        dd -= t1
        dd /= t2
        
        dd = 255.0*(dd>1.0) + 255.0*dd*(dd<=1)*(dd>=0)
        imout = cv2.applyColorMap(dd.astype('uint8'), cv2.COLORMAP_JET)
        
        out_im3[:, :, 0] = out_im2
        out_im3[:, :, 1] = out_im2
        out_im3[:, :, 2] = out_im2
        cv2.putText(out_im3, dates[k] + "_{}".format(k+1), (10, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("out_th", out_im3)
        cv2.imshow("imout", imout)
        cv2.waitKey(10)
        out_im3 = np.concatenate((out_im3, imout), axis=1)
        out.write(out_im3)
if video:
    out.release()
out_image = None
