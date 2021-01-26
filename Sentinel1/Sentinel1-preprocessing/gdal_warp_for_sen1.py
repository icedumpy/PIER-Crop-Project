from osgeo import gdal, osr

path_original = r"G:\!PIER\!FROM_2TB\Complete_VV\LSVVS403_2017-2019" # Temoplate raster for geo reference
path_before_warp = r"C:\Users\PongporC\Desktop\Sen1-planned\S1A_IW_GRDH_1SDV_20200402T111243_20200402T111308_031948_03B06C_0163.data\Sigma0_VV.img" # Image to be wrapped
path_after_warp = path_before_warp.replace(".img", "_warp.tif") # Result (after using gdal.Warp)

# Load reference raster
raster_original = gdal.Open(path_original) 

# Load new raster
raster_preprocessed_raw = gdal.Open(path_before_warp)

# =============================================================================
# reproject raster (warp) option#1 use cmd
# Note: utm47 >> EPSG:32647
#       utm48 >> EPSG:32648
#       lat,lon >> EPSG:4326
# =============================================================================
#command = f"gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r near -te {raster_original.GetGeoTransform()[0]} {raster_original.GetGeoTransform()[3] + raster_original.RasterYSize*raster_original.GetGeoTransform()[5]} {raster_original.GetGeoTransform()[0] + raster_original.RasterXSize*raster_original.GetGeoTransform()[1]} {raster_original.GetGeoTransform()[3]} -te_srs EPSG:4326 -ot Float32 -of GTiff " + "{} {}".format(path_before_warp.replace('\\', '/'), path_after_warp.replace('\\', '/'))
#os.system(command)

# =============================================================================
# reproject raster (warp) option#2 use gdal.Warp
# =============================================================================
srs = osr.SpatialReference(wkt=raster_original.GetProjection())
options = gdal.WarpOptions(format="GTiff",
                 outputBounds=(raster_original.GetGeoTransform()[0], # Min x (lon)
							   raster_original.GetGeoTransform()[3] + raster_original.RasterYSize*raster_original.GetGeoTransform()[5], # Min y (lat) 
							   raster_original.GetGeoTransform()[0] + raster_original.RasterXSize*raster_original.GetGeoTransform()[1], # Max x (lon)
							   raster_original.GetGeoTransform()[3]),# Max y (lat)
                 srcSRS = srs, dstSRS = srs,
                 outputType = gdal.GDT_Float32,
                 resampleAlg = 'near'
                 )
ds = gdal.Warp(path_after_warp, path_before_warp, options=options)
del ds
#%%