import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#%%
def calculate_kl_div(df, danger_type="Flood", column="bc(t)_p25"):
    # Eliminate outlier 
    df_loss = df.loc[df["danger_type"] == danger_type, column]
    df_loss = df_loss[df_loss.between(df_loss.quantile(0.001), df_loss.quantile(0.999))]
    df_normal = df.loc[df["danger_type"].isna(), column]
    df_normal = df_normal[df_normal.between(df_normal.quantile(0.001), df_normal.quantile(0.999))]
    
    # Calculate kde
    kde_loss = gaussian_kde(df_loss)
    kde_normal = gaussian_kde(df_normal)
    
    # Get kde
    x_min = min(kde_loss.dataset.min(), kde_normal.dataset.min()) # x_min of both loss and normal
    x_max = max(kde_loss.dataset.max(), kde_normal.dataset.max()) # x_max of both loss and normal
    x = np.linspace(x_min, x_max, 200) # total samples for kde
    kde_loss = kde_loss(x) 
    kde_normal = kde_normal(x)
    
    # Normalize range(1, 2)
    y_min = min(kde_loss.min(), kde_normal.min())
    y_max = max(kde_loss.max(), kde_normal.max())
    kde_loss = (kde_loss-y_min)/(y_max-y_min)+1
    kde_normal = (kde_normal-y_min)/(y_max-y_min)+1
    
    # Calculate KL Divergence
    kl_div = entropy(pk=kde_loss, qk=kde_normal)+entropy(pk=kde_normal, qk=kde_loss)
    
    # Plot kde and show KL div
    plt.figure()
    plt.plot(x, kde_loss, label="Loss")
    plt.plot(x, kde_normal, label="Normal")
    plt.legend(loc=1)
    plt.title(f"{danger_type}\n{column}\nKL divergence: {kl_div:.4f}")
    
    return kl_div
#%%