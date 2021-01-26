import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%%
root = r"C:\Users\PongporC\Desktop\14-7-2020"
#%%
dict_img = dict()
for file in os.listdir(root):
    if "." in file:
        if not file.split("_")[0] in dict_img.keys():
            dict_img[file.split("_")[0]] = [file]
        else:
            dict_img[file.split("_")[0]].append(file)
#%%
for key in dict_img.keys():
    file = dict_img[key]
    if "img" in file[0]:
        file_img = file[0]
        file_flood = file[1]
    else:
        file_img = file[1]
        file_flood = file[0]
    
    img = cv2.imread(os.path.join(root, file_img))
    flood = cv2.imread(os.path.join(root, file_flood))      
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flood = cv2.cvtColor(flood, cv2.COLOR_BGR2RGB)
    
    height = max(img.shape[0], flood.shape[0])
    width = max(img.shape[1], flood.shape[1])
    shape = (height, width)
    
    img = cv2.resize(img, shape)
    flood = cv2.resize(flood, shape)
    
    fig = plt.figure(figsize=(20, 10))
    ax = [fig.add_subplot(1, 2, i+1) for i in range(2)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(axis='both', which='both', length=0)
    fig.subplots_adjust(wspace=0, hspace=0)    
    
    ax[0].imshow(img)
    ax[1].imshow(flood)
    
    fig.savefig(os.path.join(r"C:\Users\PongporC\Desktop\14-7-2020\Combine", f"{key}.png"), dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)