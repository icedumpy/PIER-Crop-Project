import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icedumpy.df_tools import load_s1_flood_nonflood_dataframe
#%%
sat_type = "S1AB"
p = None

root_save = r"C:\Users\PongporC\Desktop\hist"
root_df_flood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"
root_df_nonflood = r"F:\CROP-PIER\CROP-WORK\Sentinel1_dataframe_updated"

root_df_flood = os.path.join(root_df_flood, f"{sat_type.lower()}_flood_pixel")
root_df_nonflood = os.path.join(root_df_nonflood, f"{sat_type.lower()}_nonflood_pixel")
#%%
def replace_zero_with_nan(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r"^t\d+$")]

    df.loc[:, columns_pixel_values] = df.loc[:, columns_pixel_values].replace(0, np.nan)
    return df

def add_change_columns(df):
    columns_pixel_values = df.columns[df.columns.str.contains(r'^t\d+$')].tolist()
    columns_diff = [f"{columns_pixel_values[i]}-{columns_pixel_values[i+1]}" for i in range(len(columns_pixel_values)-1)]
    df = df.assign(**{column:values for column, values in zip(columns_diff, -np.diff(df.loc[:, columns_pixel_values]).T)})
    return df, columns_pixel_values, columns_diff
#%%
for strip_id in ["302", "303", "304", "305", "306", "401", "402", "403"]:
    print(strip_id)
    try:
        # Load data
        df_flood = load_s1_flood_nonflood_dataframe(root_df=root_df_flood, p=p, strip_id=strip_id)
        df_nonflood = load_s1_flood_nonflood_dataframe(root_df_nonflood, p, strip_id=strip_id)
        
        # Replace negative data with nan
        df_flood = replace_zero_with_nan(df_flood)
        df_nonflood = replace_zero_with_nan(df_nonflood)
        
        # Select only high loss ratio (only flood)
        df_flood = df_flood[(df_flood['loss_ratio'] >= 0.8) & (df_flood['loss_ratio'] <= 1.0)]
        
        # Add change value columns (new-old)
        df_flood, columns_pixel_values, columns_diff = add_change_columns(df_flood)
        df_nonflood, columns_pixel_values, columns_diff = add_change_columns(df_nonflood)
        
        # Drop row with all nan
        df_flood = df_flood[(~df_flood[columns_pixel_values].isna().all(axis=1)) | (~df_flood[columns_diff].isna().all(axis=1))]
        df_nonflood = df_nonflood[(~df_nonflood[columns_pixel_values].isna().all(axis=1)) | (~df_nonflood[columns_diff].isna().all(axis=1))]
        
        # Sample df_nonflood
        df_nonflood = df_nonflood.sample(n=len(df_flood))
        
        fig1 = plt.figure(figsize=(16, 9))
        ax1 = fig1.gca()
        df_flood[columns_pixel_values].clip(0, 0.3).hist(bins="auto", ax=ax1)
        fig1.suptitle(f'Strip_id: ({strip_id})\nFlood pixel histogram over 8 periods', fontsize=16)
        fig1.savefig(os.path.join(root_save, f"{strip_id}_flood_pixel_hist.png"))
        
        fig2 = plt.figure(figsize=(16, 9))
        ax2 = fig2.gca()
        df_nonflood[columns_pixel_values].clip(0, 0.3).hist(bins="auto", ax=ax2)
        fig2.suptitle(f'Strip_id: ({strip_id})\nnon-Flood pixel histogram over 8 periods', fontsize=16)
        fig2.savefig(os.path.join(root_save, f"{strip_id}_nonflood_pixel_hist.png"))

        fig3 = plt.figure(figsize=(16, 9))
        ax3 = fig3.gca()
        df_flood[columns_pixel_values].min(axis=1).clip(0, 0.3).hist(bins="auto", ax=ax3)
        fig3.suptitle(f'Strip_id: ({strip_id})\nmin Flood pixel histogram over 8 periods', fontsize=16)
        fig3.savefig(os.path.join(root_save, f"{strip_id}_min_flood_pixel_hist(t-4, t+4).png"))
        
        fig4 = plt.figure(figsize=(16, 9))
        ax4 = fig4.gca()
        df_nonflood[columns_pixel_values].min(axis=1).clip(0, 0.3).hist(bins="auto", ax=ax4)
        fig4.suptitle(f'Strip_id: ({strip_id})\nmin non-Flood pixel histogram over 8 periods', fontsize=16)
        fig4.savefig(os.path.join(root_save, f"{strip_id}_min_nonflood_pixel_hist(t-4, t+4).png"))
        
        fig5 = plt.figure(figsize=(16, 9))
        ax5 = fig4.gca()
        df_flood[["t3, t4, t5"]].min(axis=1).clip(0, 0.3).hist(bins="auto", ax=ax5)
        fig5.suptitle(f'Strip_id: ({strip_id})\nmin Flood pixel histogram over 8 periods', fontsize=16)
        fig5.savefig(os.path.join(root_save, f"{strip_id}_min_flood_pixel_hist(t-2, t+1).png"))
        
        fig6 = plt.figure(figsize=(16, 9))
        ax6 = fig6.gca()
        df_nonflood[columns_pixel_values].min(axis=1).clip(0, 0.3).hist(bins="auto", ax=ax6)
        fig6.suptitle(f'Strip_id: ({strip_id})\nmin non-Flood pixel histogram over 8 periods', fontsize=16)
        fig6.savefig(os.path.join(root_save, f"{strip_id}_min_nonflood_pixel_hist(t-2, t+1).png"))        
        
        plt.close("all")
    except:
        continue


        

















