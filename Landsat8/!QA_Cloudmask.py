import rasterio
import numpy as np
import matplotlib.pyplot as plt

def extract_bits(img, position):
    """
    Extract specific bit(s)

    Parameters
    ----------
    img: numpy array (M, N)
        QA image.
    position: tuple(int, int) or int
        Bit(s)'s position read from Left to Right (if tuple)
    
    Examples
    --------
    >>> extract_bits(qa_img, position=(6, 5)) # For cloud confidence
    
    Returns
    -------
    bit
    """    
    if type(position) is tuple:
        bit_length = position[0]-position[1]+1
        bit_mask = int(bit_length*"1", 2)
        return ((img>>position[1]) & bit_mask).astype(np.uint8)
    
    elif type(position) is int:
        return ((img>>position) & 1).astype(np.uint8)
    
class LS8_QA:
    # Channel: Raster band (1, 2, 3, 4, ..., N)
    def __init__(self, raster, channel=1):
        self.raster = raster
        self.img = raster.read(channel)
        
        # Shift bits for cloud, cloud confidence, cloud shadow confidence, cirrus confidence
        # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con
        
        '''
        Cloud
        0:No 
        1:Yes
        '''
        self.cloud = extract_bits(self.img, 4)
        
        '''
        Cloud confidence, Cloud shadow confidence, Snow confidence, Cirrus confidence
        00(0):Not Determined
        01(1):Low
        10(2):Medium
        11(3):High
        '''
        self.cloud_confidence = extract_bits(self.img, (6, 5)) 
        self.cloud_shadow_confidence = extract_bits(self.img, (8, 7)) 
        self.snow_confidence = extract_bits(self.img, (10, 9))
        self.cirrus_confidence = extract_bits(self.img, (12, 11))
    

if __name__ == "__main__":
    path_raster = r"G:\!PIER\!FROM_2TB\LS8\129049_2017\ls8_129049_20170517\LC08_L1GT_129049_20170517_20170517_01_RT_BQA.TIF"
    raster = rasterio.open(path_raster)
    QA = LS8_QA(raster, channel=1)
    
    cloudmask = QA.cloud.astype('bool') | (QA.snow_confidence==3) | (QA.cloud_shadow_confidence==3)
    plt.imshow(cloudmask)