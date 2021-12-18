import numpy as np 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
#%%
def makeSamples(df_cluster):
    # Define not_related
    df_data_dict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_3c.xlsx")
    related = df_data_dict.loc[df_data_dict["training_feature_f"] == "Y", "column_nm"].tolist()
    x = df_cluster[related].values
    names = related
    y_loss = df_cluster.loss_ratio
    y_bin = (y_loss > 0.0).astype('int')
    return x, y_bin, names 
#%% Load df
df_data_dict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_3c.xlsx")
related = df_data_dict.loc[df_data_dict["training_feature_f"] == "Y", "column_nm"].tolist()

df = pd.read_pickle(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\df_pierxda_batch_3c_NE3_compressed.pkl")
df = df[~df[related].isna().any(axis=1)]
df.head()
#%%
x_dat, y_bin, names = makeSamples(df)
print(x_dat.shape)
#%%
y_bin = y_bin.values
id0 = np.nonzero(y_bin == 0)[0]
id1 = np.nonzero(y_bin == 1)[0]
num1 = len(y_bin[y_bin == 1])
num0 = len(y_bin[y_bin == 0])
print(num1, num0)
#%%
num_samples = int(0.40*num1)
id0 = np.random.permutation(id0)#[:num_samples]
id1 = np.random.permutation(id1)#[:num_samples]
id01 = id0[:num_samples]
id11 = id1[:num_samples]
id02 = id0[num_samples:2*num_samples]
id12 = id1[num_samples:2*num_samples]

x_dat2 = np.concatenate((x_dat[id11], x_dat[id01]))
y_dat2 = np.concatenate((y_bin[id11], y_bin[id01]))
x_dat3 = np.concatenate((x_dat[id12], x_dat[id02]))
y_dat3 = np.concatenate((y_bin[id12], y_bin[id02]))
print(x_dat2.shape)
print(y_dat2.mean(),y_dat3.mean())
#%%
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 
clf = RandomForestClassifier(max_depth=5, criterion="entropy", n_jobs=-1) #tree.DecisionTreeClassifier(criterion="entropy", max_depth=10)
selector_rfe = RFE(clf, n_features_to_select=30, step=1) 
# Elimination one feature at atime until we have 30 features.
pipeline = Pipeline([("sel", selector_rfe), ("rf", clf)], verbose=True) # construct a pipeline
pipeline.fit(x_dat2, y_dat2)
print(f"Train Score: {pipeline.score(x_dat2, y_dat2)}")
print(f"Test Score: {pipeline.score(x_dat3, y_dat3)}")
#%%
rank_idx = np.argsort(selector_rfe.ranking_)
important_features = []
for k in rank_idx[:30]:
  print(f"Feature{k}: {names[rank_idx[k]]}")
  important_features.append(names[rank_idx[k]])
#%%
def buildNewDataFrame(data_frame, filter_features):
    df_data_dict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_3c.xlsx")
    pass_through_features = df_data_dict.loc[df_data_dict["training_feature_f"] == "N", "column_nm"].tolist()
    data_dict ={}
    for key in data_frame.columns:
      if (key in pass_through_features) or (key in filter_features):
        data_dict[key] = data_frame[key]
    data_frame_out = pd.DataFrame(data_dict)
    return data_frame_out
#%%
df_with_important_features = buildNewDataFrame(df, important_features)
df_with_important_features.head()
#%%
df_loc = df[["ext_act_id", "polygon"]]
df_loc.head()
#%%
df_loc.polygon= gpd.GeoSeries.from_wkt(df_loc.polygon) #make it a geopandas
df_loc = gpd.GeoDataFrame(df_loc, geometry="polygon")
df_loc['center'] = df_loc.centroid # find center of all polygones
df_loc['x_loc'] = df_loc.center.x # extract longitude
df_loc['y_loc'] = df_loc.center.y #extract lattitude
#%%
df = df_with_important_features
del df_with_important_features
#%%
# Set index row
df = df.set_index('ext_act_id') 
df.index = df.index.astype('int64')
df_loc = df_loc.set_index('ext_act_id')
#%%
#merge dataframe
df = pd.merge(df, df_loc, left_index=True, right_index=True)
# make it a geopands
df = gpd.GeoDataFrame(df)
#%%
# =============================================================================
# Build Cluster based on centriods of each field
# =============================================================================
from sklearn.cluster import Birch, KMeans,AgglomerativeClustering
from IPython.display import clear_output, display
def makeCluster(locs: np.ndarray, #points locations
                num_clusters: int, # number of subcluster per levels 
                level: int, # current level 0 for the parenet
                output: np.ndarray, # output cluster label
                indices: np.ndarray, #indices of input data to be subdivided
                level_max:int=5) :  
    """
    This function is called recursively where each level a location vector is 
    divided into num_clusters clusters using a clustering algorithm.
    The ouput cluster is assigned to output variable.
    The cluster label can be represented a number based on num_cluster. 
    For example, if there are 16 clusters with num_cluster 2.
    Cluster 5 can be written as b0101 (base 2).
    We can see that at level 1, there are two clusters. Here, Cluster 5 belongs to 
    Group b0 since the first binary digit is 0.
    At level 2, there are four clusters (00, 01,10,11). Here, Cluster 5 belongs to 
    Group b01 (1) since the first two binary digit is 01.
    At level 4, there are eight clusters (000, 001,010, 011, 100,101,110,111). 
    Here, Cluster 5 belongs to Group b010(2) since the first three binary 
    digit is 010.
    """
    clustering = KMeans(n_clusters=num_clusters) # initialize a clustering algorithm
    labels = clustering.fit_predict(locs)  #perform clustering
    my_labels = labels*(num_clusters**(level_max - level-1)) + output[indices]
    output[indices] = my_labels   # assign a custer number  
    
    for lb in np.unique(labels): # for each unique label id. 
      c_locs = locs[labels == lb]   # filter only points with label lb.
      clear_output(wait=True) #clear a line before printing out.
      d = 6371 # earth radious
      s1 = (c_locs[:,0].max() - c_locs[:,0].min()) * d * np.pi / 180.0 # cluster width in meter
      s2 = (c_locs[:,1].max() - c_locs[:,1].min()) * d * np.pi / 180.0 # cluster height in meter 
      my_lb = my_labels[labels==lb]  # extact only custer with labels == lb
      progress = 100*output.max()/(num_clusters**level_max) # compute progress
      print(f"Progress [{progress:0.1f}%]: Cluster {my_lb[0]} with {len(c_locs)} locations from Level {level + 1} dx: {s1:0.2f}, dy: {s2:0.2f} km.")
      if (level + 1 < level_max) and (len(c_locs) >= num_clusters): 
        # if level is less than the max level and we have more data to divide
        idx = indices[np.nonzero(labels==lb)[0]] # obtain indeces of those with 
        # the label lb
        makeCluster(
            c_locs, num_clusters, level + 1, output, idx, 
            level_max=level_max
        ) # divide into sub clusters.
      else:
        pass
        # make subclusters. 
#%%
# extract the location in lattitude and longitude 
x = df.x_loc 
y = df.y_loc 
locs = np.stack((x,y)).T
max_levels = 12 # there are at maximum 4096 clusters
output2 = np.zeros((len(locs), ),'int') #initialized cluster ouput
indices = np.arange(len(locs)) #initialized indices
makeCluster(
    locs, # location of all clusters
    num_clusters=2, # divide into 2 subclusters at a time
    level=0, #Starting with level 0
    output=output2, #array store the cluster number
    indices=indices, #array containing indices of field to be divided.
    level_max=max_levels
) 
#%%
df[f'cluster_label'] = output2 # copy the label into a Cluster_lable column
df = df.drop(columns=["polygon", "center"])
df.to_parquet(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\cluster-level-features.parquet")
#%%
# =============================================================================
# Explore The appropriate Cluster Level.
# =============================================================================
from scipy import stats 
from IPython.display import clear_output, display
def makeSummaryDataFramesByYear(data_frame:pd.DataFrame, 
                                assignments:np.ndarray)->pd.DataFrame:
    """
    This function is to group input dataframe into a new group based on 
    assignment variable and clutivting year. 
    """
    ignore_list = [
        "polygon", "center","ext_act_id",
        "total_actual_plant_area_in_wa", "tambon_pcode","final_plant_date",
        "cluster_id", "photo_sensitive_f", "jasmine_rice_f", 
        "sticky_rice_f", "rice_age_days", "loss_type_class",  
        "cluster", 'x_loc', 'y_loc', 'y',"prov_cd", "test_f"
    ] # Field that we do not consider
    
    df_cluster = None # start with empty dataframe
    for year in np.unique(data_frame.final_plant_year): # For all clutivating years
      losses = [] 
      df_year = data_frame[data_frame.final_plant_year==year]  # Filter for a a given year
      assignments_year = assignments[data_frame.final_plant_year==year] # Extract assigment only on a field of a year of interest.
      for code in np.unique(assignments_year): # for each unique assignment values
        df_code = df_year[assignments_year == code]   # filter only the code that we are interested
        # Make sure that the cultivation starts on either May or June since 90% of the fields start here
        df_code = df_code[(df_code.final_plant_year==year) & ((df_code.final_plant_month==5) | (df_code.final_plant_month==6))]
        clear_output(wait=True) #clear a line before printing out.
        if len(df_code) > 1: # make sure that it has a least  1 field
          print(f"Combine Cluster {code} on {year} with {len(df_code)} fields.")
          data = dict()
          try:
            for key in df_code: # for each code
              if key in ignore_list: # if it is in the ignore_list discard it
                pass
              elif key == "total_actual_plant_area_in_wa":
                data[key] = df_code[key].sum() # combine all clutivation area (Do not use)
              elif key == "final_plant_year": 
                data[key] = year  # assign to a year of interst
              elif key == "final_plant_month":
                data[key] = 5 # assign all field to May
              elif key == "cluster_label":
                data[key] = code # assign the new class label into code
              elif key == "loss_ratio":
                data[key] = df_code[key].mean() # Average loss_ratio
              elif key == "loss_ratio_class":
                data[key] = stats.mode(df_code[key])[0] # pick the majority loss type
              elif key == "danger_type":
                aa = np.nonzero(df_code[key].values)[0] # pick the majoirty danger type
                if len(aa) > 0: # make sure that we get it from the loss field only.
                  data[key] = stats.mode(df_code.iloc[aa][key])[0][0]
                else:
                  data[key] = None
              elif key == "loss_type_class":
                aa = np.nonzero(df_code[key].values)[0]# pick the majoirty loss type
                if len(aa) > 0: # make sure that we get it from the loss field only.
                  data[key] = stats.mode(df_code.iloc[aa][key])[0]
                else:
                  data[key] = 0 
              elif key == "y_cls_drought_other":
                aa = np.nonzero(df_code[key].values)[0]# pick the majoirty loss value
                if len(aa) > 0: # make sure that we get it from the loss field only.
                  data[key] = stats.mode(df_code.iloc[aa][key])[0]
                else:
                  data[key] = 0 
              elif key == "y_cls_flood":
                aa = np.nonzero(df_code[key].values)[0]# pick the majoirty loss value
                if len(aa) > 0: # make sure that we get it from the loss field only.
                  data[key] = stats.mode(df_code.iloc[aa][key])[0]
                else:
                  data[key] = 0      
              else:
                # for other keys 
                # We extract min, max, median, 25 percentile and 75 percentile
                data[f"{key}_min"] = df_code[key].min()
                data[f"{key}_max"] = df_code[key].max()
                data[f"{key}_med"] = df_code[key].median()
                data[f"{key}_p25"] = df_code[key].quantile(0.25)
                data[f"{key}_75"] = df_code[key].quantile(0.75)
                #data[key] = df_code[key].median()          
            if df_cluster is None:
              df_cluster = pd.DataFrame([data])
            else:
              df2 = pd.DataFrame([data])
              df_cluster = df_cluster.append(df2, ignore_index=True)
          except Exception:
            print(key)
            break
    return df_cluster
#%%
def makeTrainTestClusterSamples(data_frame: pd.DataFrame,
                                level: int,
                                base:int = 2,
                                train_size:float = 0.8)->list:
    """
    This function divides the dataframe into a set of dataframe based on 
    their cluster labels.
    To do that, we need to extract the first level digits of the cluster_lable.
    For example, if we want a level 2 fom the maximum level of 4 with base 2.
    Cluster_label = 6 can be written as a binary number as 0110. The level 2 cluster
    is the first 2 digits which are 01. 
    Output: List of Train and test dataframes
    """
    max_level = int(np.ceil(np.log(data_frame.cluster_label.max())/np.log(base))) 
    # extract the maximum level of division 
    denor = base**(max_level - level - 1) # extract the denominator
    labels = np.floor(df['cluster_label']/denor).astype('int') 
    # extract the first level digits of the cluster label.
    num_samples = len(np.unique(labels))  # extract how many clusters in the given level.
    print(f"There are {num_samples} labels from {labels.max() +1}.")
    # randomly remap the labels into a unique number from  0 to num_samples
    lb2 = np.zeros((labels.max() + 1,),'int') #make an mapping function 
    # where inputs are the unique labels and output are random numbers
    # from 0 to num_samples.
    cnt = 0
    random_idx = np.random.permutation(num_samples)
    lbu = np.unique(labels) # get the unique cluster label value
    for k in range(labels.max() + 1):
        if k in lbu: # if we can find a cluster with label k
            lb2[k] = random_idx[cnt] # assign to cnt
            cnt += 1
        else:
            lb2[k] = num_samples + 1 # assign to -1  
    pruned_labels = lb2[labels] # randomly assign labels into a new values.
    num_train = int(train_size * num_samples)  # 80% training 
    df_train = df[pruned_labels < num_train] # assign one with ranomly assign number
    # less than num_train to train samples
    df_test = df[pruned_labels>= num_train] # assign one with ranomly assign number
    # greater than or equal num_train to train samples
    print(f"There are {len(df_train)} fields in the train dataset and {len(df_test)} fields in the test dataset")
    train_labels = labels[pruned_labels  < num_train]
    test_labels = labels[pruned_labels  >= num_train]
    for lb_test in test_labels: # check to make sure that there is no the same cluster in 
    # both train and test samples
        if lb_test in train_labels:
            print(f"We found Cluster {lb_test} in both train and test samples,")
    print(np.unique(train_labels))
    print(np.unique(test_labels))
    df_cluster_train = makeSummaryDataFramesByYear(df_train, train_labels)
    df_cluster_test = makeSummaryDataFramesByYear(df_test, test_labels)
    return df_cluster_train, df_cluster_test   
#%%
from sklearn import metrics
def makeAccuracyReport(
        x_train: np.ndarray, 
                       y_train: np.ndarray, 
                       x_test: np.ndarray, 
                       y_test: np.ndarray, 
                       clf: object, 
                       main_label:str ="", 
                       plot_output:bool=False):
    """
    This function make the accuracy report. 
    x_train: numpy array of train features
    y_train: numpy array of train labels
    x_test: numpy array of test features
    y_test: numpy array of test labels
    clf: object of classification method from sklearn
    main_label: string of plot title subfix
    plot_output: True if we want to plot the roc, precision-recall curves
    return:
     maximum f1 scores for test, and train samples. 
    
    """
    y_prob_train = clf.predict_proba(x_train) # extract predict probability on train samples
    y_prob_test = clf.predict_proba(x_test)  # extract predict probability on test samples
    k = 1  
    fpr1, tpr1, thresholds = metrics.roc_curve((y_test==k).astype('int'), y_prob_test[:,k], pos_label=1)  
    auc_score1 =metrics.auc(fpr1, tpr1) 
    
    fpr2, tpr2, thresholds = metrics.roc_curve((y_train==k).astype('int'), y_prob_train[:,k], pos_label=1)  
    auc_score2 =metrics.auc(fpr2, tpr2)
    if plot_output:
        plt.figure(figsize=(15,10))
        plt.plot(fpr1,tpr1, label=f"Test Set with AUC = {auc_score1:0.2f}")
        plt.plot(fpr2,tpr2, label=f"Train Set with AUC = {auc_score2:0.2f}")
        plt.legend()
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Detection Rate")
        plt.title("ROC" + main_label)
        plt.grid()
    target_names = ['No Loss', 'Some Loss']
    y_true = y_train
    y_pred = clf.predict(x_train)
    print("Train Accuracy in Tambon level")
    print(metrics.classification_report(y_true, y_pred, target_names=target_names))
    y_true = y_test
    y_pred = clf.predict(x_test)
    print("Test Accuracy in Tambon level")
    print(metrics.classification_report(y_true, y_pred, target_names=target_names))
    precision1, recall1, thresholds1 = metrics.precision_recall_curve((y_test==k).astype('int'), y_prob_test[:,k])
    precision2, recall2, thresholds1 = metrics.precision_recall_curve((y_train==k).astype('int'), y_prob_train[:,k], pos_label=1) 
    f11 = 2*(precision1 * recall1)/(precision1 + recall1)
    f12 = 2*(precision2 * recall2)/(precision2 + recall2)
    if plot_output:
        plt.figure(figsize=(15,10))
        plt.plot(recall1, precision1, label="Test")
        plt.plot(recall2, precision2, label="Train")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Precision and Recall Plot "+ main_label)
        plt.grid()
        plt.legend()
    print(f"Test Peak F1: {f11.max()}")
    print(f"Train Peak F1: {f12.max()}")
    #plt.grid()
    return f11.max(), f12.max()
#%%
def makeSamples(df_cluster: pd.DataFrame)->tuple:
    """
    This function extract the dataframe feature and make the numpy array.
    The output are:
    x: numpy array of features 
    y_bin: numpy array of output class where 0 and 1 correspond to no loss and some loss.
    names: List of feature names. 
    """
    df_data_dict = pd.read_excel(r"F:\CROP-PIER\CROP-WORK\20211207-PIERxDA-batch_3c-NE3\data_dict_PIERxDA_batch_3c.xlsx")
    not_related = df_data_dict.loc[df_data_dict["training_feature_f"] == "N", "column_nm"].tolist()
    not_related = not_related + ["polygon", "center", "x_loc", "y_loc"]
    related = [column for column in df_cluster.columns if not column in not_related]
    x = df_cluster[related].values
    names = related
    y_loss = df_cluster.loss_ratio
    y_bin = (y_loss > 0.0).astype('int') # if there is any loss field in the area,
    # we declare a whole cluster as loss.
    names = np.array(names)
    return x, y_bin, names
#%%
#%%
def extractSubCluster(df, df_clusters, level, max_level, based_value):
    labels = df_clusters.cluster_label
    diff_level = max_level - level 
    factor = np.arange(0, based_value**diff_level)
    label_values  = []
    for label in labels:
      possible_values = list(label*(based_value ** diff_level ) + factor)
      label_values += possible_values
    df_out = df[df.cluster_label.isin(label_values)]
    return df_out
#%%
from sklearn.feature_selection import RFE # feature selection
from sklearn.pipeline import Pipeline # link feature selector with classifier
from sklearn.ensemble import RandomForestClassifier
results = []
datasets = dict()
df_copy = df.copy()
for level in [8, 9, 10, 11]:  
      df_cluster_train, df_cluster_test = makeTrainTestClusterSamples(df_copy, level)
      x_train, y_train, names = makeSamples(df_cluster_train)
      x_test, y_test, names = makeSamples(df_cluster_test)
      datasets[level] = (x_train, y_train, x_test, y_test, names)
      df_copy = extractSubCluster(
          df, df_cluster_train, level=level, max_level=11, 
          based_value=2
      )
#%%
num_features  = [30, 40, 50] #[37, 151, 151, 151]
classifiers  = {}
for level, num_feat in zip(datasets, num_features):
    print(f"At level: {level}")
    clf = RandomForestClassifier(max_depth=5, criterion="entropy", n_jobs=-1)
    selector_rfe = RFE(clf, n_features_to_select=num_feat , step=1) 
    pipeline = Pipeline([("sel", selector_rfe), ("rf", clf)]) 
    pipeline.fit(x_train, y_train)
    score_train = pipeline.score(x_train, y_train)
    score_test = pipeline.score(x_test, y_test)
    print(f"...... with {num_feat}. Train Score: {score_train}")
    print(f"...... with {num_feat}. Test Score: {score_test}")
    print(f"...... with {num_feat}. Diff Score: {score_train - score_test}")
    classifiers[level] = pipeline
#%%
clf2 = RandomForestClassifier(max_depth=5, criterion="entropy", n_jobs=-1)
clf2.fit(x_train, y_train)
makeAccuracyReport(x_train, y_train, x_test, y_test, clf2, plot_output=True)
#%%
rank_idx = np.argsort(selector_rfe.ranking_)
for k, name in enumerate(names[rank_idx[:30]]):
  print(f"Feature{k}: {names[rank_idx[k]]}")
#%%
makeAccuracyReport(x_train, y_train, x_test, y_test, pipeline,plot_output=True)
#%%
# Add the classification result into the test data set
y_label = pipeline.predict(x_train)
df_cluster_train["clf_result"] = y_label
#%%
# extract only the field that are classified as diaster
df_train_out1 = df_cluster_train[df_cluster_train.clf_result==1]
max_level = 11
level = 10
base = 2
denor = base**(max_level - level - 1)
df_train_field = df[(df.cluster_label / denor).astype('int').isin(df_train_out1.cluster_label)] 
#%%











