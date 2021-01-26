import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread

label_values = {"TP" : 1,
                "TN" : 2,
                "FP" : 3,
                "FN" : 4}

label_color = {"TP" : "blue",
               "TN" : "green",
               "FP" : "red",
               "FN" : "yellow"}

def discrete_matshow(data):
    #get discrete colormap
    fig = plt.figure(figsize=(16, 9))

    cmap = plt.get_cmap('jet', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data)-.5, vmax=np.max(data)+.5, fignum=1)
    #tell the colorbar to tick at integers
    plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data)+1))
    return fig
    
def load_label_img(path):
    label_img = imread(path)
    return label_img

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print ('x = %d, y = %d'%(int(ix), int(iy)))

    global coords
    coords.append((int(ix), int(iy)))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
#%%
root = r"F:\CROP-PIER\CROP-WORK\Model visualization\model_127048_127049_128048_128049_B2_B3_B4_B5\127049"
for file in os.listdir(root):
    dir_img = os.path.join(root, file)
    print(dir_img)

    # Load img
    img = [imread(os.path.join(dir_img, file)) for file in os.listdir(dir_img)[:4]]
    
    # Load and show label
    label_img = load_label_img(os.path.join(dir_img, "label.png"))
    _, counts = np.unique(label_img, return_counts=True)
    print(list(label_values.keys()))
    print(counts[1:])
    
    window_size = 50
    stride = 40
    for y in range(0, label_img.shape[0]-window_size+1, stride):
        for x in range(0, label_img.shape[1]-window_size+1, stride):
            selected_x_start = x
            selected_y_start = y
            selected_x_stop = x+window_size
            selected_y_stop = y+window_size
            if (label_img[selected_y_start:selected_y_stop, selected_x_start:selected_x_stop]==0).all():
                continue
            if (label_img[selected_y_start:selected_y_stop, selected_x_start:selected_x_stop]!=0).sum()<30:
                continue
            
            plt.close("all")
            fig, ax = plt.subplots(figsize=(16, 9))
    
            stack_img = np.vstack((np.hstack((img[0][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :],
                                              img[1][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :])),
                                   np.hstack((img[2][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :],
                                              img[3][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :]))))     
            ax.imshow(stack_img)
            
            legends = []
            for key in label_values.keys():
                location = np.where(label_img[selected_y_start:selected_y_stop, selected_x_start:selected_x_stop]==label_values[key])
                rows, cols = location
                for i, (row, col) in enumerate(zip(rows, cols)):
                    if i == 0:
                        legends.append(Rectangle((col-0.5, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none', label=key))
                    ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none'))    
                    ax.add_patch(Rectangle((col-0.5, row-0.5+stack_img.shape[0]//2), 1, 1, edgecolor=label_color[key], facecolor='none'))    
                    ax.add_patch(Rectangle((col-0.5+stack_img.shape[1]//2, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none'))    
                    ax.add_patch(Rectangle((col-0.5+stack_img.shape[1]//2, row-0.5+stack_img.shape[0]//2), 1, 1, edgecolor=label_color[key], facecolor='none'))    
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Pathrow:" + dir_img.split('\\')[-2] + ", Date:"+ os.path.basename(dir_img) + f", x[{selected_x_start}:{selected_x_stop}], y[{selected_y_start}:{selected_y_stop}]")
            ax.legend(handles=legends, loc="best")
            
            dir_save = os.path.join(r"C:\Users\PongporC\Desktop\9-7-2020", dir_img.split('\\')[-2], os.path.basename(dir_img))
            os.makedirs(dir_save, exist_ok=True)
            path_save = os.path.join(dir_save, f"{len(np.nonzero(label_img[selected_y_start:selected_y_stop, selected_x_start:selected_x_stop])[0])}_{1+len(os.listdir(dir_save))}.png")
            plt.savefig(path_save, transparent=True, dpi=300)
#%% Manually select
#plt.close("all")
#coords = []
#fig = discrete_matshow(label_img)
#cid = fig.canvas.mpl_connect('key_press_event', onclick)
#%% 
#(selected_x_start, selected_y_start), (selected_x_stop, selected_y_stop) = coords
#stack_img = np.vstack((np.hstack((img[0][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :],
#                                  img[1][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :])),
#                       np.hstack((img[2][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :],
#                                  img[3][selected_y_start:selected_y_stop, selected_x_start:selected_x_stop, :]))))     
#ax.imshow(stack_img)
#
#legends = []
#for key in label_values.keys():
#    location = np.where(label_img[selected_y_start:selected_y_stop, selected_x_start:selected_x_stop]==label_values[key])
#    rows, cols = location
#    for i, (row, col) in enumerate(zip(rows, cols)):
#        if i == 0:
#            legends.append(Rectangle((col-0.5, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none', label=key))
#        ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none'))    
#        ax.add_patch(Rectangle((col-0.5, row-0.5+stack_img.shape[0]//2), 1, 1, edgecolor=label_color[key], facecolor='none'))    
#        ax.add_patch(Rectangle((col-0.5+stack_img.shape[1]//2, row-0.5), 1, 1, edgecolor=label_color[key], facecolor='none'))    
#        ax.add_patch(Rectangle((col-0.5+stack_img.shape[1]//2, row-0.5+stack_img.shape[0]//2), 1, 1, edgecolor=label_color[key], facecolor='none'))    
#
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_title("Pathrow:" + dir_img.split('\\')[-2] + ", Date:"+ os.path.basename(dir_img) + f", x[{selected_x_start}:{selected_x_stop}], y[{selected_y_start}:{selected_y_stop}]")
#ax.legend(handles=legends, loc="best")
#
#dir_save = os.path.join(r"C:\Users\PongporC\Desktop\9-7-2020", dir_img.split('\\')[-2])
#os.makedirs(dir_save, exist_ok=True)
#path_save = os.path.join(r"C:\Users\PongporC\Desktop\9-7-2020", dir_img.split('\\')[-2], f"{1+len(os.listdir(dir_save))}.png")
#plt.savefig(path_save, transparent=True, dpi=300)
#%%
window_size = 25
stride = 10
for x in range(0, 110-window_size+1, stride):
    print(x, x+window_size)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        