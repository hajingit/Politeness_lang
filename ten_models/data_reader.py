import pandas as pd
import pickle
import random
import numpy as np

def _read(feature_path="train_features.pkl", label_path="train.csv", filter_mode="", filter_val=0):
    with open(feature_path, "rb") as fin:
        pkl = pickle.load(fin)
    df = pd.read_csv(label_path, index_col=0)
    X = pkl.toarray()
    Y = df.loc[:, "ch_mean":"en_mean"].values
    arr = np.concatenate((X, Y), axis=1)
    
    if filter_mode == "value":
        print("filtering data by absolute value")
        arr = arr[(np.abs(arr[:,-2])>=filter_val) & (np.abs(arr[:,-1])>=filter_val)]
    elif filter_mode == "percentile":
        print("filtering data by percentile")
        assert filter_val % 1 == 0 and 0 < filter_val < 100, "invalid percentile value"
        if filter_val > 50:
            filter_val = 100 - filter_val
        lower1 = np.percentile(arr[:, -1], filter_val)
        upper1 = np.percentile(arr[:, -1], 100-filter_val)
        lower2 = np.percentile(arr[:, -2], filter_val)
        upper2 = np.percentile(arr[:, -2], 100-filter_val)
        
        arr = arr[np.logical_and(np.logical_or(arr[:,-1]>=upper1, arr[:,-1]<=lower1), np.logical_or(arr[:,-2]>=upper2, arr[:,-2]<=lower2))]
    
    return arr

def read(feature_path="train_features.pkl", label_path="train.csv"):
  # arr = _read(feature_path, label_path)
  # return {
  #   "X": arr[:, :-2]
  #   "Y": arr[:, -2:]
  # }
  return _read(feature_path, label_path)

def k_fold_cross_val(feature_path="train_features.pkl", label_path="train.csv", k=10, seed=777, filter_mode="", filter_val=0):
    arr = _read(feature_path, label_path, filter_mode, filter_val)
    random.seed(seed)
    
    all_ind = list(range(len(arr)))
    random.shuffle(all_ind)
    for i in range(k):
        test_ind = set(all_ind[i::k])
        train_ind = set(all_ind) - test_ind
        test_ind = list(test_ind)
        train_ind = list(train_ind)
        #print(test_ind[:10])
        yield arr[train_ind], arr[test_ind]